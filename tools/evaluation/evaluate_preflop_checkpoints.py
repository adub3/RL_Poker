import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "tools" / "evaluation" / "evaluate_strategy.py"


def checkpoint_label(path):
    name = path.name
    return name.replace("mccfr_table_", "").replace(".json.gz", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints/preflop_100bb")
    parser.add_argument("--output-dir", default="eval/preflop_100bb")
    parser.add_argument("--hands", type=int, default=1_000)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "call_check", "tight_passive"],
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("mccfr_table_iter_*.json.gz"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    for checkpoint in checkpoints:
        for opponent in args.opponents:
            output_dir = Path(args.output_dir) / checkpoint_label(checkpoint) / opponent
            command = [
                sys.executable,
                str(EVALUATOR),
                "--checkpoint",
                str(checkpoint),
                "--opponent",
                opponent,
                "--hands",
                str(args.hands),
                "--output-dir",
                str(output_dir),
            ]
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
