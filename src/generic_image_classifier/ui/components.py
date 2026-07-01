from __future__ import annotations

from pathlib import Path

import gradio as gr

from generic_image_classifier.data.zip import (
    count_classes_in_zip,
    validate_class_zip,
)
from generic_image_classifier.device import resolve_device
from generic_image_classifier.logging import get_logger
from generic_image_classifier.models import predict_image
from generic_image_classifier.schemas import Model
from generic_image_classifier.ui.theme import MODEL_CATALOG

logger = get_logger(__name__)

EMPTY_CLASS_TABLE = "| Class | Images (approx.) |\n| --- | ---: |\n"
EXAMPLE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def panel(title: str) -> gr.Markdown:
    return gr.Markdown(f'<div class="gic-panel"><h3>{title}</h3></div>')


def hero_block(title: str, subtitle: str, steps: list[str]) -> gr.Markdown:
    step_html = "".join(f'<span class="gic-step">{s}</span>' for s in steps)
    return gr.Markdown(
        f"""
        <div class="gic-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="gic-steps">{step_html}</div>
        </div>
        """
    )


def model_blurb(model_name: str) -> str:
    label, detail = MODEL_CATALOG.get(model_name, (model_name, ""))
    return f"**{label}** — {detail}" if detail else f"**{label}**"


def preview_zip_contents(zip_file) -> tuple[str, str, bool]:
    if zip_file is None:
        return (
            '<div class="gic-banner-warn">Upload a ZIP with one folder per class.</div>',
            EMPTY_CLASS_TABLE,
            False,
        )

    error = validate_class_zip(zip_file.name)
    if error:
        banner = f'<div class="gic-banner-warn">{error}</div>'
        return banner, "", False

    counts = count_classes_in_zip(zip_file.name)
    rows = "\n".join(f"| {cls} | {n} |" for cls, n in sorted(counts.items()))
    table = EMPTY_CLASS_TABLE + rows
    total = sum(counts.values())
    banner = (
        f'<div class="gic-banner-ok">Found <strong>{len(counts)}</strong> classes '
        f"and <strong>{total}</strong> images. Ready to train.</div>"
    )
    return banner, table, True


def metrics_markdown(schema: Model) -> str:
    info = schema.info
    results = schema.results
    acc = f"{results.accuracy:.1%}" if results else "—"
    f1 = f"{results.f1_score:.3f}" if results and results.f1_score is not None else "—"
    precision = f"{results.precision:.3f}" if results and results.precision is not None else "—"
    recall = f"{results.recall:.3f}" if results and results.recall is not None else "—"
    epochs = schema.history.epochs_trained

    cards = [
        ("Accuracy", acc),
        ("F1", f1),
        ("Precision", precision),
        ("Recall", recall),
        ("Classes", str(info.num_classes)),
        ("Epochs", str(epochs)),
    ]
    parts = [
        f'<div class="gic-metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in cards
    ]
    return f'<div class="gic-metric-grid">{"".join(parts)}</div>'


def class_metrics_dataframe(schema: Model):
    import pandas as pd

    if not schema.results or not schema.results.class_accuracies:
        return pd.DataFrame(columns=["Class", "Precision"])
    rows = [
        {"Class": name, "Precision": f"{score:.1%}"}
        for name, score in sorted(
            schema.results.class_accuracies.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return pd.DataFrame(rows)


def model_info_markdown(schema: Model, device_name: str) -> str:
    info = schema.info
    h, w, c = info.input_shape
    return f"""
| | |
| --- | --- |
| **Architecture** | `{info.name}` |
| **Classes** | {info.num_classes} |
| **Input size** | {h} × {w} × {c} |
| **Parameters** | {info.params_count:,} |
| **Trained** | {info.date_trained} |
| **Device** | {device_name} |
"""


def run_classification(image, model, schema: Model | None, device):
    import pandas as pd

    if image is None:
        return "Drop or upload an image to classify.", None, None, ""

    if model is None or schema is None:
        return (
            "Train a model first, or load a checkpoint via the CLI.",
            None,
            None,
            '<div class="gic-banner-warn">No model loaded yet.</div>',
        )

    img_size = (schema.info.input_shape[0], schema.info.input_shape[1])
    try:
        label, confidence, probs = predict_image(
            model,
            image,
            img_size,
            schema.info.class_mapping,
            device=device,
        )
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        df = pd.DataFrame([{"Class": k, "Confidence": f"{v:.1%}"} for k, v in sorted_probs.items()])
        banner = (
            f'<div class="gic-banner-ok">Top prediction: '
            f"<strong>{label}</strong> ({confidence:.1%})</div>"
        )
        return f"{label} — {confidence:.1%}", sorted_probs, df, banner
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return f"Prediction error: {exc}", None, None, ""


def device_label() -> str:
    dev = resolve_device()
    return "GPU (CUDA)" if dev.type == "cuda" else "CPU"


def list_example_images(examples_dir: Path | None) -> list[str]:
    if not examples_dir or not examples_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(examples_dir.iterdir())
        if path.suffix.lower() in EXAMPLE_SUFFIXES
    ]


def example_gallery_items(paths: list[str]) -> list[tuple[str, str]]:
    return [(path, Path(path).stem) for path in paths]
