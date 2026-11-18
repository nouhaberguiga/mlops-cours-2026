"""Pipeline Kubeflow complet pour le scoring crédit."""

from kfp import compiler, dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Artifact, Dataset, Input, Model, Output


PIPELINE_IMAGE = "localhost:5001/scoring-pipeline:1.0.0"


@dsl.component(base_image=PIPELINE_IMAGE)
def ingest_data(raw_data: Output[Dataset]):
    from pipelines.stage import run_stage

    run_stage(
        "src.download",
        outputs={"data/raw/dataset.csv": raw_data.path},
    )


@dsl.component(base_image=PIPELINE_IMAGE)
def validate_data(
    raw_data: Input[Dataset],
    validation_report: Output[Artifact],
):
    from pipelines.stage import run_stage

    run_stage(
        "src.validate_data",
        inputs={"data/raw/dataset.csv": raw_data.path},
        outputs={"reports/validation.json": validation_report.path},
    )


@dsl.component(base_image=PIPELINE_IMAGE)
def prepare_features(
    raw_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    features_data: Output[Dataset],
):
    from pipelines.stage import run_stage

    run_stage(
        "src.prepare",
        inputs={"data/raw/dataset.csv": raw_data.path},
        outputs={
            "data/processed/train.csv": train_data.path,
            "data/processed/test.csv": test_data.path,
            "data/processed/features.parquet": features_data.path,
        },
    )


@dsl.component(base_image=PIPELINE_IMAGE)
def train_model(
    raw_data: Input[Dataset],
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    mlflow_uri: str,
    kfp_run_id: str,
):
    import os

    from mlflow import MlflowClient
    from mlflow.models import Model as MLflowModel
    from pipelines.stage import run_stage

    run_stage(
        "src.train",
        inputs={
            "data/raw/dataset.csv": raw_data.path,
            "data/processed/train.csv": train_data.path,
            "data/processed/test.csv": test_data.path,
        },
        outputs={"models/model": model.path},
        env={
            "MLFLOW_TRACKING_URI": mlflow_uri,
            "LAUNCHED_BY": "kfp",
        },
    )

    run_id = MLflowModel.load(model.path).metadata["mlflow_run_id"]
    client = MlflowClient(tracking_uri=mlflow_uri)
    client.set_tag(run_id, "launched_by", "kfp")
    client.set_tag(run_id, "kfp_run_id", kfp_run_id)

    model.metadata["mlflow_run_id"] = run_id
    print(f"Modèle entraîné. MLflow run_id : {run_id}")


@dsl.component(base_image=PIPELINE_IMAGE)
def evaluate_model(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Input[Model],
    metrics_report: Output[Artifact],
    mlflow_uri: str,
):
    from pipelines.stage import run_stage

    run_stage(
        "src.evaluate",
        inputs={
            "data/processed/train.csv": train_data.path,
            "data/processed/test.csv": test_data.path,
            "models/model": model.path,
        },
        outputs={"reports/metrics.json": metrics_report.path},
        env={"MLFLOW_TRACKING_URI": mlflow_uri},
    )


@dsl.component(base_image=PIPELINE_IMAGE)
def register_model(
    model: Input[Model],
    registration_report: Output[Artifact],
    mlflow_uri: str,
):
    from pathlib import Path

    from mlflow.models import Model as MLflowModel
    from pipelines.stage import run_stage

    run_id = MLflowModel.load(model.path).metadata["mlflow_run_id"]

    run_stage(
        "src.promote",
        args=["--run-id", run_id],
        env={"MLFLOW_TRACKING_URI": mlflow_uri},
    )

    Path(registration_report.path).parent.mkdir(parents=True, exist_ok=True)
    Path(registration_report.path).write_text(
        f"Modèle scoring-model enregistré pour le run {run_id}\n",
        encoding="utf-8",
    )


@dsl.pipeline(
    name="scoring-pipeline",
    description="Pipeline de scoring crédit : ingestion, validation, préparation, entraînement, évaluation et registre.",
)
def scoring_pipeline(
    mlflow_uri: str = "http://mlflow.mlflow:5000",
):
    ingest = ingest_data()
    ingest.set_retry(2)
    ingest.set_memory_request("512Mi")
    ingest.set_memory_limit("1Gi")
    kubernetes.set_timeout(ingest, 600)

    validate = validate_data(raw_data=ingest.outputs["raw_data"])
    validate.set_retry(2)
    validate.set_memory_request("512Mi")
    validate.set_memory_limit("1Gi")
    kubernetes.set_timeout(validate, 600)

    prepare = prepare_features(raw_data=ingest.outputs["raw_data"])
    prepare.set_retry(2)
    prepare.set_memory_request("1Gi")
    prepare.set_memory_limit("2Gi")
    kubernetes.set_timeout(prepare, 900)

    train = train_model(
        raw_data=ingest.outputs["raw_data"],
        train_data=prepare.outputs["train_data"],
        test_data=prepare.outputs["test_data"],
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    train.set_retry(2)
    train.set_memory_request("2Gi")
    train.set_memory_limit("3Gi")
    train.set_cpu_request("500m")
    train.set_caching_options(enable_caching=False)
    kubernetes.set_timeout(train, 1800)

    evaluate = evaluate_model(
        train_data=prepare.outputs["train_data"],
        test_data=prepare.outputs["test_data"],
        model=train.outputs["model"],
        mlflow_uri=mlflow_uri,
    )
    evaluate.set_retry(2)
    evaluate.set_memory_request("1Gi")
    evaluate.set_memory_limit("2Gi")
    kubernetes.set_timeout(evaluate, 900)

    register = register_model(
        model=train.outputs["model"],
        mlflow_uri=mlflow_uri,
    )
    register.after(evaluate)
    register.set_memory_request("512Mi")
    register.set_memory_limit("1Gi")
    register.set_caching_options(enable_caching=False)
    kubernetes.set_timeout(register, 600)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=scoring_pipeline,
        package_path="scoring_pipeline.yaml",
    )
    print("Pipeline compilé : scoring_pipeline.yaml")
