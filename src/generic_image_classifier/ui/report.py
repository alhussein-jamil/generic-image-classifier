from __future__ import annotations

import pandas as pd

from generic_image_classifier.schemas import Model
from generic_image_classifier.ui.components import (
    class_metrics_dataframe,
    device_label,
    metrics_markdown,
    model_info_markdown,
)
from generic_image_classifier.ui.plots import plot_confusion_matrix, plot_training_history


def empty_class_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["Class", "Precision"])


def report_view(schema: Model | None) -> tuple[str, str, str, pd.DataFrame, object, object]:
    if schema is None:
        return (
            "Train a model to populate metrics and per-class performance.",
            "",
            "",
            empty_class_table(),
            None,
            None,
        )
    return (
        "",
        metrics_markdown(schema),
        model_info_markdown(schema, device_label()),
        class_metrics_dataframe(schema),
        plot_training_history(schema),
        plot_confusion_matrix(schema),
    )
