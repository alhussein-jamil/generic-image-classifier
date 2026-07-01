import pytest
import torch
from PIL import Image

from generic_image_classifier.data.transforms import build_transforms
from generic_image_classifier.models.backbones import create_model
from generic_image_classifier.schemas import ModelConfig


def test_build_transforms_output_shape():
    transform = build_transforms((32, 32), augment=False)
    image = Image.new("RGB", (64, 64), color="red")
    tensor = transform(image)
    assert tensor.shape == (3, 32, 32)


@pytest.mark.parametrize("model_name", ["mobilenetv2", "resnet50"])
def test_create_model_forward(model_name: str):
    config = ModelConfig(model_name=model_name, num_classes=3, pretrained=False)
    model = create_model(config, input_shape=(32, 32, 3))
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (2, 3)
