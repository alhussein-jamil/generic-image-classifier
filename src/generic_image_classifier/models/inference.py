import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from generic_image_classifier.data.transforms import load_image_tensor
from generic_image_classifier.device import resolve_device
from generic_image_classifier.logging import get_logger
from generic_image_classifier.models.backbones import create_model
from generic_image_classifier.schemas import Model, ModelConfig

logger = get_logger(__name__)


def load_model_schema(checkpoint_dir: Path, model_name: str) -> Model | None:
    schema_path = checkpoint_dir / f"{model_name}_schema.json"
    if not schema_path.exists():
        logger.warning("Schema not found: %s", schema_path)
        return None
    with schema_path.open() as f:
        return Model.from_dict(json.load(f))


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, Model]:
    device = device or resolve_device()
    schema_path = (
        checkpoint_path.parent / f"{checkpoint_path.stem.replace('_best', '')}_schema.json"
    )
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema required alongside checkpoint: {schema_path}")

    with schema_path.open() as f:
        model_schema = Model.from_dict(json.load(f))
    info = model_schema.info

    model_config = ModelConfig(
        model_name=info.name,
        num_classes=info.num_classes,
        pretrained=False,
    )
    model = create_model(model_config, tuple(info.input_shape))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded %s (%d classes)", info.name, info.num_classes)
    return model, model_schema


def predict_image(
    model: nn.Module,
    img_path: Path | str,
    img_size: tuple[int, int],
    class_mapping: dict[int, str],
    device: torch.device | None = None,
) -> tuple[str, float, dict[str, float]]:
    device = device or resolve_device()
    model.eval()

    tensor = load_image_tensor(img_path, img_size, device)
    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0]

    predicted_id = probs.argmax().item()
    confidence = probs[predicted_id].item()
    predicted_class = class_mapping.get(predicted_id, f"Unknown ({predicted_id})")
    all_predictions = {
        class_mapping.get(i, f"Class {i}"): prob.item() for i, prob in enumerate(probs)
    }
    return predicted_class, confidence, all_predictions
