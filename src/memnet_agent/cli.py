from __future__ import annotations

import argparse
import json
from pathlib import Path

from .memory import AssociativeMemory
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memnet-agent", description="Manage memnet-agent graphs")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show graph statistics and validation status")
    info.add_argument("graph", type=Path)

    validate = subparsers.add_parser("validate", help="Validate a graph")
    validate.add_argument("graph", type=Path)

    convert = subparsers.add_parser("convert", help="Convert a graph to SQLite, JSON, GraphML or zip")
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--format", default="auto")

    dataset = subparsers.add_parser("dataset", help="Generate a JSONL training dataset")
    dataset.add_argument("graph", type=Path)
    dataset.add_argument("destination", type=Path)
    dataset.add_argument("--iterations", type=int, default=1)
    dataset.add_argument("--max-examples", type=int, default=2000)
    dataset.add_argument("--format", choices=("chat", "instruction", "graph"), default="chat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    memory = AssociativeMemory.load_external(args.graph if hasattr(args, "graph") else args.source)

    if args.command == "info":
        print(json.dumps({"stats": memory.stats(), "validation_errors": memory.validate()}, indent=2))
        return
    if args.command == "validate":
        errors = memory.validate()
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            raise SystemExit(1)
        print(json.dumps({"valid": True, "errors": []}, indent=2))
        return
    if args.command == "convert":
        destination = memory.export(args.destination, format=args.format)
        print(destination)
        return
    if args.command == "dataset":
        from .dataset import write_training_dataset

        destination = write_training_dataset(
            memory,
            args.destination,
            iterations=args.iterations,
            max_examples_per_iteration=args.max_examples,
            format=args.format,
        )
        print(destination)
        return


if __name__ == "__main__":
    main()
