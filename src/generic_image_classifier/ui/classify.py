from __future__ import annotations

import gradio as gr
import torch

from generic_image_classifier.schemas import Model
from generic_image_classifier.ui.components import run_classification


def pick_gallery_image(evt: gr.SelectData):
    value = evt.value
    if isinstance(value, dict):
        return value.get("image", value)
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def wire_classify_tab(
    *,
    device: torch.device,
    image_input: gr.Image,
    classify_btn: gr.Button,
    clear_btn: gr.Button,
    result_text: gr.Textbox,
    class_probs: gr.Label,
    prob_table: gr.Dataframe,
    result_banner: gr.Markdown,
    model_holder: gr.State,
    schema_holder: gr.State,
    gallery: gr.Gallery | None = None,
) -> None:
    def classify(image, model, schema):
        return run_classification(image, model, schema, device)

    outputs = [result_text, class_probs, prob_table, result_banner]
    inputs = [image_input, model_holder, schema_holder]

    classify_btn.click(classify, inputs=inputs, outputs=outputs)
    image_input.change(classify, inputs=inputs, outputs=outputs)
    clear_btn.click(
        lambda: (None, "", None, None, ""),
        outputs=[image_input, result_text, class_probs, prob_table, result_banner],
    )
    if gallery is not None:
        gallery.select(pick_gallery_image, outputs=image_input)


def wire_classify_tab_loaded(
    *,
    device: torch.device,
    model,
    schema: Model,
    image_input: gr.Image,
    submit_btn: gr.Button,
    clear_btn: gr.Button,
    result_text: gr.Textbox,
    label_output: gr.Label,
    prob_table: gr.Dataframe,
    result_banner: gr.Markdown,
) -> None:
    def classify(image):
        return run_classification(image, model, schema, device)

    outputs = [result_text, label_output, prob_table, result_banner]
    submit_btn.click(classify, inputs=[image_input], outputs=outputs)
    image_input.change(classify, inputs=[image_input], outputs=outputs)
    clear_btn.click(
        lambda: (None, "", None, None, ""),
        outputs=[image_input, result_text, label_output, prob_table, result_banner],
    )
