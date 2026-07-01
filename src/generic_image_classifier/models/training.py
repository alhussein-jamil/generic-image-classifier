import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from generic_image_classifier.device import resolve_device
from generic_image_classifier.logging import get_logger
from generic_image_classifier.models.optim import build_criterion, build_optimizer
from generic_image_classifier.schemas import (
    Dataset,
    Model,
    ModelConfig,
    ModelInfo,
    ModelResults,
    TrainingHistory,
)

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: ModelConfig,
    dataset: Dataset,
    checkpoint_dir: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, Model]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = model.to(device)
    criterion = build_criterion(config.loss)
    optimizer = build_optimizer(config.optimizer, model.parameters(), config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = TrainingHistory()
    best_val_accuracy = 0.0
    best_model_state: dict | None = None
    best_checkpoint = checkpoint_dir / f"{config.model_name}_best.pt"

    logger.info("Training for %d epochs on %s", config.epochs, device)
    for epoch in range(config.epochs):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, train=False
        )
        scheduler.step(val_loss)
        history.update(
            {
                "accuracy": train_acc,
                "val_accuracy": val_acc,
                "loss": train_loss,
                "val_loss": val_loss,
            }
        )
        logger.info(
            "Epoch %d/%d  train %.4f/%.4f  val %.4f/%.4f",
            epoch + 1,
            config.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "model_name": config.model_name,
                    "num_classes": config.num_classes,
                },
                best_checkpoint,
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    eval_results = evaluate_model(model, val_loader, dataset.metadata.inv_class_mapping, device)

    sample_batch, _ = next(iter(train_loader))
    input_height, input_width = sample_batch.shape[2], sample_batch.shape[3]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_schema = Model(
        info=ModelInfo(
            name=config.model_name,
            num_classes=config.num_classes,
            input_shape=(input_height, input_width, 3),
            class_mapping=dataset.metadata.inv_class_mapping,
            date_trained=timestamp,
            params_count=sum(p.numel() for p in model.parameters()),
        ),
        history=history,
        results=eval_results,
        checkpoint_path=best_checkpoint,
    )

    schema_path = checkpoint_dir / f"{config.model_name}_schema.json"
    with schema_path.open("w") as f:
        json.dump(model_schema.to_dict(), f, indent=2)
    logger.info("Saved schema to %s", schema_path)

    return model, model_schema


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0
    desc = "Training" if train else "Evaluating"
    ctx = torch.enable_grad() if train else torch.inference_mode()

    with ctx:
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_mapping: dict[int, str],
    device: torch.device,
) -> ModelResults:
    model.eval()
    true_labels: list[int] = []
    predictions: list[int] = []

    with torch.inference_mode():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            preds = model(inputs).argmax(1)
            true_labels.extend(targets.tolist())
            predictions.extend(preds.cpu().tolist())

    true_arr = np.array(true_labels)
    pred_arr = np.array(predictions)
    cm = confusion_matrix(true_arr, pred_arr)
    report = classification_report(true_arr, pred_arr, output_dict=True)

    class_accuracies = {
        class_mapping[class_id]: report[str(class_id)]["precision"]
        for class_id in class_mapping
        if str(class_id) in report
    }

    results = ModelResults(
        accuracy=report["accuracy"],
        class_accuracies=class_accuracies,
        confusion_matrix=cm.tolist(),
        f1_score=report["macro avg"]["f1-score"],
        precision=report["macro avg"]["precision"],
        recall=report["macro avg"]["recall"],
    )
    logger.info("Accuracy %.4f  F1 %.4f", results.accuracy, results.f1_score or 0)
    return results
