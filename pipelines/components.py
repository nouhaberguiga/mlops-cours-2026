"""Composants KFP partagés par les orchestrateurs."""

from pipelines.scoring_pipeline import (
    PIPELINE_IMAGE,
    ingest_data,
    validate_data,
    prepare_features,
    train_model,
    evaluate_model,
    register_model,
)

__all__ = [
    "PIPELINE_IMAGE",
    "ingest_data",
    "validate_data",
    "prepare_features",
    "train_model",
    "evaluate_model",
    "register_model",
]
