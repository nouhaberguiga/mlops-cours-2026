"""Pipeline de démonstration du Data Drift."""

from kfp import dsl
import kfp.kubernetes as kubernetes

from pipelines.components import (
    PIPELINE_IMAGE,
    ingest_data,
    validate_data,
    prepare_features,
    detect_drift,
)
from pipelines.scoring_pipeline import simulate_drift
from pipelines.scoring_pipeline import DRIFT


@dsl.pipeline(
    name="scoring-drift-demo",
    description="Démontre la détection du drift sans bloquer le pipeline.",
)
def drift_pipeline():
    ingest = ingest_data()
    ingest.set_memory_request("512Mi")
    ingest.set_memory_limit("1Gi")

    validate = validate_data(
        raw_data=ingest.outputs["raw_data"],
    )
    validate.set_memory_request("512Mi")
    validate.set_memory_limit("1Gi")

    prepare = prepare_features(
        raw_data=ingest.outputs["raw_data"],
    )
    prepare.after(validate)
    prepare.set_memory_request("1Gi")
    prepare.set_memory_limit("2Gi")
    kubernetes.set_timeout(prepare, 900)

    simulated = simulate_drift(
        test_data=prepare.outputs["test_data"],
    )
    simulated.set_memory_request("512Mi")
    simulated.set_memory_limit("1Gi")
    kubernetes.set_timeout(simulated, 600)

    drift = detect_drift(
        reference_data=prepare.outputs["train_data"],
        current_data=simulated.outputs["drift_data"],
        ks_alpha=float(DRIFT["ks_alpha"]),
        psi_threshold=float(DRIFT["psi_threshold"]),
        drift_share=float(DRIFT["drift_share"]),
        fail_on_drift=False,
    )
    drift.set_memory_request("1Gi")
    drift.set_memory_limit("2Gi")
    kubernetes.set_timeout(drift, 900)
