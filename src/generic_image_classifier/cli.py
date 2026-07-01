import argparse
import sys
from pathlib import Path

from generic_image_classifier.config import (
    create_default_config,
    create_interactive_config,
    load_config,
    save_config,
    update_config,
)
from generic_image_classifier.logging import get_logger, setup_logging
from generic_image_classifier.pipeline import run_training
from generic_image_classifier.schemas import Config
from generic_image_classifier.ui import launch_app

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image classification toolkit")
    parser.add_argument(
        "--mode",
        choices=["train", "app", "both"],
        default="app",
        help="train, app, or both (default: app)",
    )
    parser.add_argument("--zip", type=str, help="Dataset zip path")
    parser.add_argument("--extract_dir", type=str, help="Extraction directory")
    parser.add_argument("--model", type=str, default="mobilenetv2", help="Backbone name")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--img_size", type=int, nargs=2, help="Image height width")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--title", type=str, help="App title")
    parser.add_argument("--config", type=str, help="Config JSON path")
    parser.add_argument("--save_config", type=str, help="Save config to path")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch UI without a dataset (default: on)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config: Config | None = None

    if args.config:
        config = load_config(Path(args.config))
        if config is None:
            logger.error("Could not load config: %s", args.config)
            sys.exit(1)
    elif args.zip:
        config = create_default_config(Path(args.zip))
    elif args.interactive:
        config = create_interactive_config(port=args.port)
    else:
        logger.error("Provide --zip, --config, or use --interactive")
        sys.exit(1)

    updates: dict = {}
    if args.extract_dir:
        updates["data.extract_dir"] = Path(args.extract_dir)
    if args.model:
        updates["model.model_name"] = args.model
    if args.epochs:
        updates["model.epochs"] = args.epochs
    if args.batch_size:
        updates["data.batch_size"] = args.batch_size
    if args.lr:
        updates["model.learning_rate"] = args.lr
    if args.img_size:
        updates["data.img_size"] = tuple(args.img_size)
    if args.port:
        updates["app.port"] = args.port
    if args.title:
        updates["app.title"] = args.title
    if updates:
        config = update_config(config, updates)

    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)
    if args.device:
        config.device = args.device

    if args.save_config:
        save_config(config, Path(args.save_config))

    return config


def main() -> None:
    setup_logging()
    args = parse_args()
    config = build_config(args)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name: str | None = None
    if args.mode in ("train", "both"):
        model_name, _, _ = run_training(config)
    if args.mode in ("app", "both"):
        launch_app(config.app, config.checkpoint_dir, model_name)


if __name__ == "__main__":
    main()
