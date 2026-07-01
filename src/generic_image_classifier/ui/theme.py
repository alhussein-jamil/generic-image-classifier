import gradio as gr

CUSTOM_CSS = """
:root {
    --gic-radius: 12px;
    --gic-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
}

.gradio-container {
    max-width: 1180px !important;
}

.gic-hero {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 45%, #134e4a 100%);
    border-radius: var(--gic-radius);
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    color: #f0fdfa;
    box-shadow: var(--gic-shadow);
}

.gic-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.gic-hero p {
    margin: 0;
    opacity: 0.92;
    line-height: 1.5;
}

.gic-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 1.1rem;
}

.gic-step {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 600;
}

.gic-panel {
    border: 1px solid #e2e8f0;
    border-radius: var(--gic-radius);
    padding: 1rem 1.1rem;
    background: #ffffff;
    box-shadow: var(--gic-shadow);
}

.gic-panel h3 {
    margin: 0 0 0.75rem 0;
    font-size: 1rem;
    color: #0f172a;
}

.gic-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 0.5rem 0 1rem 0;
}

.gic-metric {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    background: #f8fafc;
}

.gic-metric .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.25rem;
}

.gic-metric .value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f766e;
    line-height: 1.2;
}

.gic-hint {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.45;
}

.gic-banner-warn {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #92400e;
    margin-bottom: 0.75rem;
}

.gic-banner-ok {
    border-left: 4px solid #0f766e;
    background: #f0fdfa;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #115e59;
    margin-bottom: 0.75rem;
}

footer {
    opacity: 0.65;
}
"""


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#f1f5f9",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e2e8f0",
        block_radius="12px",
        block_shadow="0 8px 30px rgba(15, 23, 42, 0.06)",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        input_background_fill="#f8fafc",
    )


MODEL_CATALOG: dict[str, tuple[str, str]] = {
    "mobilenetv2": ("MobileNetV2", "Lightweight. Good first pick for quick runs."),
    "resnet50": ("ResNet-50", "Strong baseline with moderate training cost."),
    "densenet121": ("DenseNet-121", "Efficient feature reuse. Works well on smaller sets."),
    "efficientnetb0": ("EfficientNet-B0", "Balanced accuracy per parameter."),
    "vgg16": ("VGG-16", "Classic architecture. Slower but interpretable."),
}

MODEL_CHOICES = list(MODEL_CATALOG.keys())
