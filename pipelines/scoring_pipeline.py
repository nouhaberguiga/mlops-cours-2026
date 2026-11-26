"""Pipeline Kubeflow complet pour le scoring crédit."""

from kfp import compiler, dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Artifact, Dataset, HTML, Input, Markdown, Metrics, Model, Output
from pathlib import Path
import yaml


PIPELINE_IMAGE = "localhost:5001/scoring-pipeline:1.3.0"
PARAMS = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
TRAIN = PARAMS["train"]
DRIFT = PARAMS["drift"]


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
    validation_html: Output[HTML],
):
    from pipelines.stage import run_stage

    run_stage(
        "src.validate_data",
        inputs={"data/raw/dataset.csv": raw_data.path},
        outputs={
            "reports/validation.json": validation_report.path,
            "reports/validation.html": validation_html.path,
        },
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
def simulate_drift(
    test_data: Input[Dataset],
    drift_data: Output[Dataset],
):
    """Génère un jeu de données avec un drift simulé."""
    from pipelines.stage import run_stage

    run_stage(
        "src.simulate",
        args=[
            "drift",
            "data/processed/test.csv",
            "data/drift/test_drift.csv",
        ],
        inputs={
            "data/processed/test.csv": test_data.path,
        },
        outputs={
            "data/drift/test_drift.csv": drift_data.path,
        },
    )

@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def detect_drift(
    reference_data: Input[Dataset],
    current_data: Input[Dataset],
    report: Output[Artifact],
    table: Output[Markdown],
    html: Output[HTML],
    drift: Output[Metrics],
    ks_alpha: float,
    psi_threshold: float,
    drift_share: float,
    fail_on_drift: bool,
) -> bool:
    """Détecte le Data Drift entre les données de référence et les données courantes."""
    import json
    from pathlib import Path

    from pipelines.stage import run_stage
    from src.detect_drift import REPORT_HTML, REPORT_JSON, REPORT_MD

    reference = "data/drift/reference.csv"
    current = "data/drift/test_drift.csv"

    args = [
        reference,
        current,
        "--ks-alpha", ks_alpha,
        "--psi-threshold", psi_threshold,
        "--drift-share", drift_share,
    ]

    if not fail_on_drift:
        args.append("--no-fail")

    run_stage(
        "src.detect_drift",
        args=args,
        inputs={
            reference: reference_data.path,
            current: current_data.path,
        },
        outputs={
            str(REPORT_JSON): report.path,
            str(REPORT_MD): table.path,
            str(REPORT_HTML): html.path,
        },
    )

    summary = json.loads(Path(report.path).read_text(encoding="utf-8"))

    drift.log_metric("n_columns", summary["n_columns"])
    drift.log_metric("n_drifted", summary["n_drifted"])
    drift.log_metric("share_drifted", summary["share_drifted"])
    drift.log_metric(
        "evidently_drifted_columns",
        summary["evidently"]["drifted_columns"],
    )

    return bool(summary["drift_detected"])


@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def train_model(
    raw_data: Input[Dataset],
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    model_type: str,
    C: float,
    class_weight: str,
    n_estimators: int,
    max_depth: str,
    learning_rate: float,
    random_state: int,
    mlflow_uri: str,
    kfp_run_id: str,
):
    from mlflow import MlflowClient
    from mlflow.models import Model as MLflowModel
    from pipelines.stage import run_stage

    run_stage(
        "src.train",
        args=[
            "--model", model_type,
            "--C", C,
            "--class_weight", class_weight,
            "--n_estimators", n_estimators,
            "--max_depth", max_depth,
            "--learning_rate", learning_rate,
            "--random_state", random_state,
        ],
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

@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def validate_model(
    model: Input[Model],
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    report: Output[Artifact],
    table: Output[Markdown],
    html: Output[HTML],
    checks: Output[Metrics],
):
    """Valide le modèle avec Evidently et les tests comportementaux."""
    import json
    from pathlib import Path

    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import MODEL_DIR, TEST_CSV, TRAIN_CSV
    from src.validate_model import REPORT_HTML, REPORT_JSON, REPORT_MD

    cfg = load_config()

    run_stage(
        "src.validate_model",
        inputs={
            str(MODEL_DIR): model.path,
            str(TRAIN_CSV): train_data.path,
            str(TEST_CSV): test_data.path,
        },
        outputs={
            str(REPORT_JSON): report.path,
            str(REPORT_MD): table.path,
            str(REPORT_HTML): html.path,
        },
    )

    summary = json.loads(Path(report.path).read_text(encoding="utf-8"))

    for name in ("n_checks", "n_failed", cfg.primary_metric):
        checks.log_metric(name, summary[name])


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
    prepare.after(validate)
    prepare.set_retry(2)
    prepare.set_memory_request("1Gi")
    prepare.set_memory_limit("2Gi")
    kubernetes.set_timeout(prepare, 900)

    drift = detect_drift(
        reference_data=prepare.outputs["train_data"],
        current_data=prepare.outputs["test_data"],
        ks_alpha=float(DRIFT["ks_alpha"]),
        psi_threshold=float(DRIFT["psi_threshold"]),
        drift_share=float(DRIFT["drift_share"]),
        fail_on_drift=True,
    )
    drift.set_memory_request("1Gi")
    drift.set_memory_limit("2Gi")
    kubernetes.set_timeout(drift, 900)

    train = train_model(
        raw_data=ingest.outputs["raw_data"],
        train_data=prepare.outputs["train_data"],
        test_data=prepare.outputs["test_data"],
        model_type=str(TRAIN["model"]),
        C=float(TRAIN["C"]),
        class_weight=str(TRAIN["class_weight"]),
        n_estimators=int(TRAIN["n_estimators"]),
        max_depth=str(TRAIN["max_depth"]),
        learning_rate=float(TRAIN["learning_rate"]),
        random_state=int(TRAIN["random_state"]),
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    train.after(drift)
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

    model_validation = validate_model(
        model=train.outputs["model"],
        train_data=prepare.outputs["train_data"],
        test_data=prepare.outputs["test_data"],
    )
    model_validation.set_memory_request("1Gi")
    model_validation.set_memory_limit("2Gi")
    kubernetes.set_timeout(model_validation, 900)

    register = register_model(
        model=train.outputs["model"],
        mlflow_uri=mlflow_uri,
    )
    register.after(evaluate, model_validation)
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
