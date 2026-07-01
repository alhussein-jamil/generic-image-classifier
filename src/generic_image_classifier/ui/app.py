from pathlib import Path

import gradio as gr

from generic_image_classifier.data import ZipUploadError, extract_uploaded_dataset
from generic_image_classifier.device import resolve_device
from generic_image_classifier.logging import get_logger
from generic_image_classifier.models import load_model_from_checkpoint, load_model_schema
from generic_image_classifier.pipeline import build_training_config, run_training
from generic_image_classifier.schemas import AppConfig
from generic_image_classifier.ui.classify import wire_classify_tab, wire_classify_tab_loaded
from generic_image_classifier.ui.components import (
    example_gallery_items,
    hero_block,
    list_example_images,
    metrics_markdown,
    model_blurb,
    panel,
    preview_zip_contents,
)
from generic_image_classifier.ui.plots import plot_confusion_matrix, plot_training_history
from generic_image_classifier.ui.report import report_view
from generic_image_classifier.ui.theme import CUSTOM_CSS, MODEL_CHOICES, build_theme

logger = get_logger(__name__)

WARN_BANNER = '<div class="gic-banner-warn">Upload a ZIP with one folder per class.</div>'


def create_all_in_one_app() -> gr.Blocks:
    device = resolve_device()

    with gr.Blocks(title="Image Classifier", fill_height=True) as app:
        hero_block(
            "Image Classifier",
            "Upload labeled images, fine-tune a backbone, and test predictions in one place.",
            ["1 · Upload dataset", "2 · Train", "3 · Classify"],
        )

        model_holder = gr.State(None)
        schema_holder = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Dataset & train", id="train_tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        panel("Dataset")
                        dataset_upload = gr.File(
                            label="Dataset ZIP",
                            file_types=[".zip"],
                            file_count="single",
                        )
                        dataset_status = gr.Markdown(WARN_BANNER)
                        class_preview = gr.Markdown(
                            value="| Class | Images (approx.) |\n| --- | ---: |\n",
                        )

                        panel("Training settings")
                        model_type = gr.Dropdown(
                            label="Backbone",
                            choices=MODEL_CHOICES,
                            value="mobilenetv2",
                        )
                        model_hint = gr.Markdown(model_blurb("mobilenetv2"))

                        with gr.Row():
                            img_height = gr.Number(
                                label="Height",
                                value=224,
                                precision=0,
                                minimum=64,
                                maximum=512,
                            )
                            img_width = gr.Number(
                                label="Width",
                                value=224,
                                precision=0,
                                minimum=64,
                                maximum=512,
                            )

                        with gr.Accordion("Advanced options", open=False):
                            batch_size = gr.Slider(8, 64, value=32, step=8, label="Batch size")
                            epochs = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                            learning_rate = gr.Slider(
                                0.0001,
                                0.01,
                                value=0.001,
                                step=0.0001,
                                label="Learning rate",
                            )
                            val_split = gr.Slider(
                                0.1,
                                0.5,
                                value=0.2,
                                step=0.05,
                                label="Validation split",
                            )
                            use_pretrained = gr.Checkbox(
                                label="ImageNet pretrained weights",
                                value=True,
                            )
                            use_augmentation = gr.Checkbox(
                                label="Training augmentation",
                                value=True,
                            )

                        train_btn = gr.Button(
                            "Start training",
                            variant="primary",
                            interactive=False,
                        )
                        gr.Markdown(
                            '<p class="gic-hint">Checkpoints land in <code>checkpoints/</code>. '
                            "Larger backbones need more VRAM.</p>"
                        )

                    with gr.Column(scale=6):
                        panel("Training results")
                        training_status = gr.Markdown(
                            "Results appear here after training completes."
                        )
                        result_metrics = gr.Markdown("")
                        history_plot = gr.Image(
                            label="Learning curves",
                            type="numpy",
                            interactive=False,
                        )
                        cm_plot = gr.Image(
                            label="Confusion matrix",
                            type="numpy",
                            interactive=False,
                        )

                def on_dataset_change(zip_file):
                    banner, table, ready = preview_zip_contents(zip_file)
                    return banner, table, gr.update(interactive=ready)

                def train_from_ui(
                    zip_file,
                    model_type,
                    img_height,
                    img_width,
                    batch_size,
                    epochs,
                    learning_rate,
                    use_pretrained,
                    val_split,
                    use_augmentation,
                    progress=gr.Progress(track_tqdm=True),  # noqa: B008
                ):
                    progress(0, desc="Preparing dataset")
                    empty = report_view(None)
                    try:
                        zip_path, extract_dir = extract_uploaded_dataset(zip_file)
                    except ZipUploadError as exc:
                        return (str(exc), "", None, None, None, None, [], *empty)

                    config = build_training_config(
                        zip_path=zip_path,
                        extract_dir=extract_dir,
                        model_name=model_type,
                        img_size=(int(img_height), int(img_width)),
                        batch_size=int(batch_size),
                        val_split=val_split,
                        augmentation=use_augmentation,
                        pretrained=use_pretrained,
                        learning_rate=learning_rate,
                        epochs=int(epochs),
                    )

                    try:
                        progress(0.05, desc="Training model")
                        _, model_schema, _ = run_training(config)
                        model, schema = load_model_from_checkpoint(
                            model_schema.checkpoint_path, device=device
                        )
                        gallery = example_gallery_items(
                            list_example_images(config.checkpoint_dir / "examples")
                        )
                        progress(1.0, desc="Done")
                        report = report_view(schema)
                        return (
                            f"Training finished. Checkpoint saved to `{config.checkpoint_dir}`.",
                            metrics_markdown(schema),
                            plot_training_history(schema),
                            plot_confusion_matrix(schema),
                            model,
                            schema,
                            gallery,
                            *report,
                        )
                    except Exception as exc:
                        logger.exception("Training failed")
                        return (f"Training failed: {exc}", "", None, None, None, None, [], *empty)

                dataset_upload.change(
                    on_dataset_change,
                    inputs=[dataset_upload],
                    outputs=[dataset_status, class_preview, train_btn],
                )
                model_type.change(
                    lambda name: model_blurb(name),
                    inputs=[model_type],
                    outputs=[model_hint],
                )

            with gr.Tab("Classify", id="classify_tab"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_input = gr.Image(
                            type="filepath",
                            label="Input image",
                            height=420,
                        )
                        with gr.Row():
                            classify_btn = gr.Button("Classify", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                        classify_examples = gr.Gallery(
                            label="Per-class examples (after training)",
                            columns=3,
                            height=220,
                            object_fit="contain",
                            interactive=False,
                        )

                    with gr.Column(scale=5):
                        result_banner = gr.Markdown("")
                        result_text = gr.Textbox(
                            label="Top label",
                            interactive=False,
                            max_lines=1,
                        )
                        class_probs = gr.Label(
                            label="Confidence by class",
                            num_top_classes=12,
                        )
                        prob_table = gr.Dataframe(
                            headers=["Class", "Confidence"],
                            label="Ranked predictions",
                            interactive=False,
                            wrap=True,
                        )

                wire_classify_tab(
                    device=device,
                    image_input=image_input,
                    classify_btn=classify_btn,
                    clear_btn=clear_btn,
                    result_text=result_text,
                    class_probs=class_probs,
                    prob_table=prob_table,
                    result_banner=result_banner,
                    model_holder=model_holder,
                    schema_holder=schema_holder,
                    gallery=classify_examples,
                )

            with gr.Tab("Model report", id="report_tab"):
                report_intro = gr.Markdown(
                    "Train a model to populate metrics and per-class performance."
                )
                report_metrics = gr.Markdown("")
                report_info = gr.Markdown("")
                report_classes = gr.Dataframe(
                    headers=["Class", "Precision"],
                    label="Per-class precision",
                    interactive=False,
                )
                with gr.Row():
                    report_history = gr.Image(
                        label="Learning curves",
                        type="numpy",
                        interactive=False,
                    )
                    report_cm = gr.Image(
                        label="Confusion matrix",
                        type="numpy",
                        interactive=False,
                    )

        train_btn.click(
            train_from_ui,
            inputs=[
                dataset_upload,
                model_type,
                img_height,
                img_width,
                batch_size,
                epochs,
                learning_rate,
                use_pretrained,
                val_split,
                use_augmentation,
            ],
            outputs=[
                training_status,
                result_metrics,
                history_plot,
                cm_plot,
                model_holder,
                schema_holder,
                classify_examples,
                report_intro,
                report_metrics,
                report_info,
                report_classes,
                report_history,
                report_cm,
            ],
            show_progress="full",
        )

    return app


def create_app(config: AppConfig, checkpoint_dir: Path, model_name: str) -> gr.Blocks:
    model_schema = load_model_schema(checkpoint_dir, model_name)
    if not model_schema or not model_schema.checkpoint_path:
        raise ValueError(f"No checkpoint found for {model_name}")

    device = resolve_device()
    model, schema = load_model_from_checkpoint(model_schema.checkpoint_path, device=device)
    examples = list_example_images(config.examples_dir)

    with gr.Blocks(title=config.title, fill_height=True) as app:
        hero_block(
            config.title,
            config.description,
            ["Upload", "Predict", "Review metrics"],
        )

        with gr.Tabs():
            with gr.Tab("Classify"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_input = gr.Image(
                            type="filepath",
                            label="Input image",
                            height=440,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Classify", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                        if examples:
                            gr.Examples(
                                examples=examples,
                                inputs=image_input,
                                label="Sample images from training set",
                            )

                    with gr.Column(scale=5):
                        result_banner = gr.Markdown("")
                        result_text = gr.Textbox(
                            label="Top label",
                            interactive=False,
                            max_lines=1,
                        )
                        label_output = gr.Label(
                            label="Confidence by class",
                            num_top_classes=12,
                        )
                        prob_table = gr.Dataframe(
                            headers=["Class", "Confidence"],
                            label="Ranked predictions",
                            interactive=False,
                            wrap=True,
                        )

                wire_classify_tab_loaded(
                    device=device,
                    model=model,
                    schema=schema,
                    image_input=image_input,
                    submit_btn=submit_btn,
                    clear_btn=clear_btn,
                    result_text=result_text,
                    label_output=label_output,
                    prob_table=prob_table,
                    result_banner=result_banner,
                )

            with gr.Tab("Model report"):
                _, metrics, info, classes, history, cm = report_view(schema)
                gr.Markdown(metrics)
                gr.Markdown(info)
                gr.Dataframe(
                    value=classes,
                    headers=["Class", "Precision"],
                    label="Per-class precision",
                    interactive=False,
                )
                with gr.Row():
                    gr.Image(
                        value=history,
                        label="Learning curves",
                        type="numpy",
                        interactive=False,
                    )
                    gr.Image(
                        value=cm,
                        label="Confusion matrix",
                        type="numpy",
                        interactive=False,
                    )

    return app


def launch_app(config: AppConfig, checkpoint_dir: Path, model_name: str | None) -> None:
    if not model_name or not checkpoint_dir.exists():
        logger.info("Launching all-in-one UI")
        app = create_all_in_one_app()
    else:
        logger.info("Launching inference UI for %s", model_name)
        app = create_app(config, checkpoint_dir, model_name)
    app.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        share=False,
        theme=build_theme(),
        css=CUSTOM_CSS,
    )
