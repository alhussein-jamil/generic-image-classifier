from generic_image_classifier.device import dataloader_workers, resolve_device


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_dataloader_workers_non_negative():
    assert dataloader_workers() >= 0
