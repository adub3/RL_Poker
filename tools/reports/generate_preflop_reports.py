import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NEW_CODE = ROOT / "new_code"
sys.path.insert(0, str(NEW_CODE))

from ai import load_table  # noqa: E402
from preflop import (  # noqa: E402
    RANKS,
    aggregate_preflop_strategy,
    empty_hand_matrix,
    entropy,
    hand_matrix_position,
    hand_matrix_to_rows,
    normalized_action_probs,
    preflop_coverage_metrics,
)


GROUPS = ("fold", "call_check", "raise", "all_in")


def action_group(action, all_in_action):
    key = str(action)
    if key in ("0", "fold"):
        return "fold"
    if key in ("1", "call", "check", "call_check"):
        return "call_check"
    if key == "jam":
        return "all_in"
    if all_in_action is not None and key == all_in_action:
        return "all_in"
    return "raise"


def grouped_probs(action_mass):
    raw_raise_actions = []
    for action in action_mass:
        try:
            raw_action = int(action)
        except ValueError:
            continue
        if raw_action > 1:
            raw_raise_actions.append(raw_action)
    all_in_action = str(max(raw_raise_actions)) if raw_raise_actions else None
    grouped = defaultdict(float)
    for action, mass in action_mass.items():
        grouped[action_group(action, all_in_action)] += float(mass)
    total = sum(grouped.values())
    if total <= 0:
        return {group: 0.0 for group in GROUPS}
    return {group: grouped[group] / total for group in GROUPS}


def write_csv(path, header, rows):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def write_matrix(path, matrix):
    write_csv(path, [""] + list(RANKS), hand_matrix_to_rows(matrix))


def build_matrices(aggregates):
    action_matrix = empty_hand_matrix("")
    raise_matrix = empty_hand_matrix("")

    for hand_class, action_mass in aggregates.items():
        row, col = hand_matrix_position(hand_class)
        probs = grouped_probs(action_mass)
        likely_action = max(GROUPS, key=lambda group: probs[group])
        raise_frequency = probs["raise"] + probs["all_in"]
        action_matrix[row][col] = likely_action
        raise_matrix[row][col] = f"{raise_frequency:.3f}"

    return action_matrix, raise_matrix


def context_matches(context, selected_contexts):
    if selected_contexts is None:
        return True
    for selected in selected_contexts:
        if selected == "open" and ("[seq:open]" in context or context == "[]"):
            return True
        if selected == context:
            return True
    return False


def context_file_label(context):
    if context == "open" or "[seq:open]" in context:
        prefix = "open"
    else:
        prefix = context.strip("[]").replace("][", "_")
    return prefix.replace("/", "_").replace(" ", "_").replace(":", "-") or "empty"


def action_sort_key(action):
    try:
        return (0, int(action))
    except ValueError:
        return (1, str(action))


def write_action_tables(output_dir, aggregates, contexts, selected_contexts=None):
    rows = []
    for hand_class in sorted(aggregates, key=lambda hand: hand_matrix_position(hand)):
        probs = grouped_probs(aggregates[hand_class])
        rows.append(
            [
                hand_class,
                f"{probs['fold']:.6f}",
                f"{probs['call_check']:.6f}",
                f"{probs['raise']:.6f}",
                f"{probs['all_in']:.6f}",
                f"{entropy(aggregates[hand_class]):.6f}",
            ]
        )
    write_csv(
        output_dir / "action_probabilities_by_hand.csv",
        ["hand", "fold", "call_check", "raise", "all_in", "entropy"],
        rows,
    )

    context_rows = []
    for hand_class in sorted(contexts, key=lambda hand: hand_matrix_position(hand)):
        for context, action_mass in sorted(contexts[hand_class].items()):
            if not context_matches(context, selected_contexts):
                continue
            probs = normalized_action_probs(action_mass)
            grouped = grouped_probs(action_mass)
            context_rows.append(
                [
                    hand_class,
                    context,
                    f"{grouped['fold']:.6f}",
                    f"{grouped['call_check']:.6f}",
                    f"{grouped['raise']:.6f}",
                    f"{grouped['all_in']:.6f}",
                    " ".join(
                        f"{action}:{prob:.4f}"
                        for action, prob in sorted(probs.items(), key=lambda item: action_sort_key(item[0]))
                    ),
                ]
            )
    write_csv(
        output_dir / "action_probabilities_by_context.csv",
        ["hand", "context", "fold", "call_check", "raise", "all_in", "raw_actions"],
        context_rows,
    )


def write_context_matrices(output_dir, contexts, selected_contexts=None):
    if selected_contexts is None:
        context_groups = {
            context: [context]
            for context in sorted(
                {
                    context
                    for hand_contexts in contexts.values()
                    for context in hand_contexts
                },
                key=lambda value: (len(value), value),
            )
        }
    else:
        context_groups = {}
        for selected in selected_contexts:
            if selected == "open":
                matches = sorted(
                    {
                        context
                        for hand_contexts in contexts.values()
                        for context in hand_contexts
                        if context_matches(context, {selected})
                    },
                    key=lambda value: (len(value), value),
                )
                context_groups[selected] = matches
            else:
                context_groups[selected] = [selected]

    for output_context, source_contexts in context_groups.items():
        if not source_contexts:
            continue

        safe_name = context_file_label(output_context)
        action_matrix = empty_hand_matrix("")
        raise_matrix = empty_hand_matrix("")

        for hand_class, hand_contexts in contexts.items():
            combined_mass = defaultdict(float)
            for source_context in source_contexts:
                for action, mass in hand_contexts.get(source_context, {}).items():
                    combined_mass[action] += mass
            if not combined_mass:
                continue
            row, col = hand_matrix_position(hand_class)
            probs = grouped_probs(combined_mass)
            likely_action = max(GROUPS, key=lambda group: probs[group])
            raise_frequency = probs["raise"] + probs["all_in"]
            action_matrix[row][col] = likely_action
            raise_matrix[row][col] = f"{raise_frequency:.3f}"

        action_path = output_dir / f"most_likely_action_matrix_{safe_name}.csv"
        raise_path = output_dir / f"raise_frequency_matrix_{safe_name}.csv"
        write_matrix(action_path, action_matrix)
        write_matrix(raise_path, raise_matrix)
        write_optional_heatmap(
            action_path,
            output_dir / f"most_likely_action_matrix_{safe_name}.png",
            f"Most Likely Preflop Action {output_context}",
        )
        write_optional_heatmap(
            raise_path,
            output_dir / f"raise_frequency_matrix_{safe_name}.png",
            f"Preflop Raise Frequency {output_context}",
            numeric=True,
        )


def write_bucket_frequency(metrics_path, output_dir):
    if not metrics_path or not metrics_path.exists():
        return
    with metrics_path.open("r") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return

    keys = [
        key for key in rows[0] if key.startswith("strategy_mass_")
    ]
    out_rows = []
    for row in rows:
        total = sum(float(row.get(key, 0.0) or 0.0) for key in keys)
        out_rows.append(
            [row.get("iteration", ""), *[
                f"{(float(row.get(key, 0.0) or 0.0) / total):.6f}" if total > 0 else "0.000000"
                for key in keys
            ]]
        )
    write_csv(
        output_dir / "bucket_frequency_over_stages.csv",
        ["iteration", *[key.replace("strategy_mass_", "") for key in keys]],
        out_rows,
    )


def write_optional_heatmap(matrix_path, output_path, title, numeric=False):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    with matrix_path.open("r") as csv_file:
        rows = list(csv.reader(csv_file))[1:]

    fig, ax = plt.subplots(figsize=(9, 8))
    if numeric:
        values = [
            [float(value) if value else 0.0 for value in row[1:]]
            for row in rows
        ]
        image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        colors = {"": 0, "fold": 1, "call_check": 2, "raise": 3, "all_in": 4}
        values = [[colors.get(value, 0) for value in row[1:]] for row in rows]
        ax.imshow(values, cmap="tab20", vmin=0, vmax=4)
        for y, row in enumerate(rows):
            for x, value in enumerate(row[1:]):
                if value:
                    ax.text(x, y, value.replace("_", "\n"), ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(RANKS)), list(RANKS))
    ax.set_yticks(range(len(RANKS)), list(RANKS))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(output_dir, checkpoint, table, aggregates):
    coverage = preflop_coverage_metrics(table)
    lines = [
        "# Preflop Report",
        "",
        f"Checkpoint: {checkpoint}",
        f"Preflop infosets: {coverage['preflop_infosets']}",
        f"Starting hand classes seen: {coverage['starting_hand_classes']} / 169",
        f"Average entropy by hand: {coverage['avg_strategy_entropy_by_hand']:.4f}",
        "",
        "Generated files:",
        "- most_likely_action_matrix.csv",
        "- raise_frequency_matrix.csv",
        "- action_probabilities_by_hand.csv",
        "- action_probabilities_by_context.csv",
    ]
    if aggregates:
        most_visited = max(aggregates, key=lambda hand: sum(aggregates[hand].values()))
        lines.extend(
            [
                "",
                f"Highest strategy-mass hand class: {most_visited}",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--output-dir", default="assets/diagrams/preflop_report")
    parser.add_argument(
        "--contexts",
        nargs="*",
        default=["open"],
        help=(
            "Optional context names to report, e.g. open or [] . "
            "Defaults to open only."
        ),
    )
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help="Write matrices/tables for every betting context. This can create many files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(args.checkpoint)
    aggregates, contexts = aggregate_preflop_strategy(table)
    action_matrix, raise_matrix = build_matrices(aggregates)
    selected_contexts = None
    if args.all_contexts:
        selected_contexts = None
    elif args.contexts is not None:
        selected_contexts = set(args.contexts)

    action_matrix_path = output_dir / "most_likely_action_matrix.csv"
    raise_matrix_path = output_dir / "raise_frequency_matrix.csv"
    write_matrix(action_matrix_path, action_matrix)
    write_matrix(raise_matrix_path, raise_matrix)
    write_action_tables(output_dir, aggregates, contexts, selected_contexts)
    write_context_matrices(output_dir, contexts, selected_contexts)
    write_bucket_frequency(Path(args.metrics) if args.metrics else None, output_dir)
    write_optional_heatmap(
        action_matrix_path,
        output_dir / "most_likely_action_matrix.png",
        "Most Likely Preflop Action",
    )
    write_optional_heatmap(
        raise_matrix_path,
        output_dir / "raise_frequency_matrix.png",
        "Preflop Raise Frequency",
        numeric=True,
    )
    write_summary(output_dir, args.checkpoint, table, aggregates)
    print(f"Wrote preflop reports to {output_dir}")


if __name__ == "__main__":
    main()
