"""Pipeline de test : vérifie que les données corrompues sont bloquées."""

from kfp import dsl
import kfp.kubernetes as kubernetes
from kfp.dsl import Dataset, Input, Output

from pipelines.components import (
    PIPELINE_IMAGE,
    ingest_data,
    validate_data,
    prepare_features,
)


@dsl.component(base_image=PIPELINE_IMAGE)
def simulate_bad_data(
    raw_data: Input[Dataset],
    corrupted_data: Output[Dataset],
):
    """Simule des erreurs silencieuses dans le dataset."""
    from pipelines.stage import run_stage

    run_stage(
        "src.simulate",
        args=[
            "corrupt",
            "data/raw/dataset.csv",
            "data/raw/dataset_corrupted.csv",
        ],
        inputs={
            "data/raw/dataset.csv": raw_data.path,
        },
        outputs={
            "data/raw/dataset_corrupted.csv": corrupted_data.path,
        },
    )


@dsl.pipeline(
    name="scoring-bad-data-test",
    description="Vérifie que le Gate 1 bloque les données corrompues.",
)
def bad_data_pipeline():
    ingest = ingest_data()
    ingest.set_memory_request("512Mi")
    ingest.set_memory_limit("1Gi")

    corrupt = simulate_bad_data(
        raw_data=ingest.outputs["raw_data"],
    )
    corrupt.set_memory_request("512Mi")
    corrupt.set_memory_limit("1Gi")

    validate = validate_data(
        raw_data=corrupt.outputs["corrupted_data"],
    )
    validate.set_memory_request("512Mi")
    validate.set_memory_limit("1Gi")

    prepare = prepare_features(
        raw_data=corrupt.outputs["corrupted_data"],
    )
    prepare.after(validate)
    prepare.set_memory_request("1Gi")
    prepare.set_memory_limit("2Gi")
    kubernetes.set_timeout(prepare, 900)
