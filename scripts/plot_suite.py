#!/usr/bin/env python3
"""Plot saved live-arm results with repeat dots; does not run model inference.

Requires matplotlib in the analysis environment. Bars average repeated runs of
identical selected questions within a conversation. Dots show every included
repeat, not independent new QA samples; no confidence interval is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_suite import saved_complete_runs

LIVE_ARMS = ("baseline", "optimized", "fused", "joint")


def plot_rows(report, origin, suite_dir=None):
    """Include completed cold arms even when a later arm or replay fails."""
    rows = []
    for cell in report["cells"]:
        runs = saved_complete_runs(cell, suite_dir)
        if not any(arm in runs for arm in LIVE_ARMS):
            continue
        order = [mode for mode in cell["modes"].split(",") if mode in LIVE_ARMS]
        for arm in LIVE_ARMS:
            if arm not in runs:
                continue
            run = runs[arm]
            rows.append({"origin": origin, "cell": cell["name"], "sample_index": cell["sample_index"],
                         "sample_id": cell["sample_id"], "turns": cell["actual_turns"],
                         "question_count": cell["question_count"], "repeat": cell["repeat"],
                         "arm": arm, "experiment_policy": cell.get("evolution_policy", "upstream"),
                         "order": order, "build_seconds": run["build_seconds"],
                         "http_requests_build": run["build_llm"]["http_requests"],
                         "token_f1": run["metrics"]["token_f1"]})
    return rows


def source_counts(cells):
    completed = sum(cell["status"] == "complete" for cell in cells)
    failed = sum(cell["status"] == "failed" for cell in cells)
    return {"planned": len(cells), "completed": completed, "failed": failed,
            "pending_or_running": len(cells) - completed - failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="SVG, PDF, or PNG output")
    parser.add_argument("--png", action="store_true", help="Also write a PNG preview next to SVG/PDF")
    parser.add_argument("--analysis", type=Path, help="Optional analyze_suite_views.py JSON")
    parser.add_argument("--view", choices=("planned_all", "planned_allocation_stable", "supplemental_balanced_stable"), default="planned_all")
    parser.add_argument("--confirmation-suite", type=Path)
    parser.add_argument("--policy", choices=("upstream", "fused", "joint"), default="upstream", help="Experiment policy if suite contains multiple policy groups")
    parser.add_argument("--turns", type=int, help="Required if suite has more than one turn count")
    parser.add_argument("--arms", nargs="+", choices=LIVE_ARMS, help="Limit figure to these cold arms")
    parser.add_argument("--paired-only", action="store_true",
                        help="Keep only cells where every selected cold arm completed")
    args = parser.parse_args()
    report = json.loads(args.suite.read_text())
    all_rows = plot_rows(report, "primary", args.suite.parent)
    source_cells = [{**cell, "origin": "primary"} for cell in report["cells"]]
    if args.confirmation_suite:
        confirmation = json.loads(args.confirmation_suite.read_text())
        all_rows += plot_rows(confirmation, "confirmation", args.confirmation_suite.parent)
        source_cells += [{**cell, "origin": "confirmation"} for cell in confirmation["cells"]]
    if args.analysis:
        analysis = json.loads(args.analysis.read_text())
        view = analysis["views"][args.view]
        if view.get("available") is False:
            parser.error(view.get("reason", "Requested view unavailable"))
        references = {(ref["origin"], ref["name"]) for ref in view["cell_refs"]}
        all_rows = [row for row in all_rows if (row["origin"], row["cell"]) in references]
    elif args.view != "planned_all":
        parser.error("Sensitivity views require --analysis")
    turn_counts = sorted({row["turns"] for row in all_rows})
    if not turn_counts:
        parser.error("No completed cold arms to plot")
    if args.turns is None and len(turn_counts) != 1:
        parser.error("Specify --turns when plotting a suite with multiple turn counts")
    selected_turns = args.turns if args.turns is not None else turn_counts[0]
    rows = [row for row in all_rows if row["turns"] == selected_turns and row["experiment_policy"] == args.policy]
    selected_arms = set(args.arms or {row["arm"] for row in rows})
    rows = [row for row in rows if row["arm"] in selected_arms]
    if args.paired_only:
        if "baseline" not in selected_arms or len(selected_arms) < 2:
            parser.error("Paired figures require baseline and at least one candidate arm")
        references = {(row["origin"], row["cell"]) for row in rows}
        complete_pairs = {reference for reference in references
                          if {row["arm"] for row in rows if (row["origin"], row["cell"]) == reference} >= selected_arms}
        rows = [row for row in rows if (row["origin"], row["cell"]) in complete_pairs]
    if not rows:
        parser.error(f"No completed cold arms with {selected_turns} turns")
    samples = sorted({row["sample_index"] for row in rows})
    json_guidance = bool(report.get("args", {}).get("structured_policy_outputs", False))
    arm_styles = [("baseline", "Configured robust", "#778899"),
                  ("optimized", {"upstream": "Exact pipeline", "fused": "Fused pipeline",
                                 "joint": "Joint pipeline"}[args.policy], "#0072B2"),
                  ("fused", "Fused evolution + schema" if json_guidance else "Fused evolution", "#D55E00"),
                  ("joint", "Joint note + schema" if json_guidance else "Joint metadata + evolution", "#009E73")]
    present_arms = {row["arm"] for row in rows}
    arms = [style for style in arm_styles if style[0] in present_arms]
    arm_counts = {arm: sum(row["arm"] == arm for row in rows) for arm, _, _ in arms}
    pair_annotations = []
    if args.paired_only:
        baselines = {(row["origin"], row["cell"]): row for row in rows if row["arm"] == "baseline"}
        for sample in samples:
            for arm, _, _ in arms:
                if arm == "baseline":
                    continue
                selected = [row for row in rows if row["sample_index"] == sample and row["arm"] == arm]
                pair_annotations.append({"sample_index": sample, "arm": arm, "paired_runs": len(selected),
                                         "build_speedup_geomean": math.exp(statistics.mean(
                                             math.log(baselines[(row["origin"], row["cell"])]["build_seconds"] / row["build_seconds"])
                                             for row in selected)),
                                         "f1_delta_mean": statistics.mean(row["token_f1"] - baselines[(row["origin"], row["cell"])]["token_f1"]
                                                                          for row in selected)})
    width = 0.76 / len(arms)
    offsets = {arm: (position - (len(arms) - 1) / 2) * width
               for position, (arm, _, _) in enumerate(arms)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "svg.fonttype": "none", "axes.spines.top": False,
                                "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.1))
    panels = [("build_seconds", "Memory build time", "Seconds"),
              ("http_requests_build", "Live requests during build", "HTTP requests"),
              ("token_f1", "QA quality", "Normalized token F1")]
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for arm, label, color in arms:
            labelled = False
            for sample_position, sample in enumerate(samples):
                selected = [row for row in rows if row["sample_index"] == sample and row["arm"] == arm]
                if not selected:
                    continue
                values = [row[metric] for row in selected]
                x = sample_position + offsets[arm]
                axis.bar(x, statistics.mean(values), width=width * 0.9, color=color,
                         alpha=0.8, label=f"{label} (n={arm_counts[arm]})" if not labelled else None, zorder=2)
                labelled = True
                for repeat_index, value in enumerate(values):
                    spread = (repeat_index - (len(values) - 1) / 2) * min(0.045, width / (len(values) + 1))
                    axis.scatter(x + spread, value, s=16, color="#111111", edgecolors="white",
                                 linewidths=0.35, zorder=3)
                if args.paired_only and arm != "baseline" and metric in {"build_seconds", "token_f1"}:
                    annotation = next(item for item in pair_annotations if item["sample_index"] == sample and item["arm"] == arm)
                    label = (f"{annotation['build_speedup_geomean']:.2f}×" if metric == "build_seconds"
                             else f"Δ {annotation['f1_delta_mean']:+.3f}")
                    if metric == "token_f1":
                        candidates = [item[0] for item in arms if item[0] != "baseline"]
                        group_max = max(row[metric] for row in rows if row["sample_index"] == sample)
                        first_slot = min(group_max + 0.045, 0.97 - 0.065 * (len(candidates) - 1))
                        label_y = first_slot + 0.065 * candidates.index(arm)
                        axis.annotate(label, (x, max(values)), xytext=(x, label_y), textcoords="data",
                                      ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
                                      arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.5},
                                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
                    else:
                        axis.annotate(label, (x, max(values)), xytext=(0, 6), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_ylabel(ylabel)
        labels = [f"Conversation {sample}" for sample in samples]
        if args.paired_only:
            labels = [label + "\nN=" + str(sum(row["arm"] == "baseline" and row["sample_index"] == sample for row in rows))
                      for label, sample in zip(labels, samples)]
        axis.set_xticks(range(len(samples)), labels)
        axis.grid(axis="y", color="#dce2e7", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        if metric == "token_f1":
            axis.set_ylim(0, 1)
        else:
            axis.margins(y=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.23),
               ncol=len(arms), frameon=False)
    pair_count = len({(row["origin"], row["cell"]) for row in rows})
    model = report.get("args", {}).get("model", "model")
    cell_label = "cell" if pair_count == 1 else "cells"
    scope = (f"{pair_count} completed selected-arm pairs" if len(arms) == 2 else f"{pair_count} completed {len(arms)}-arm sets") if args.paired_only else f"{pair_count} {cell_label} with completed arms"
    fig.suptitle(f"LoCoMo {selected_turns}-turn subsets · {model} · {scope}",
                 fontsize=14, fontweight="bold", y=0.99)
    view_label = {"planned_all": "All planned cells with completed cold arms",
                  "planned_allocation_stable": "Sensitivity: allocation-stable planned cells",
                  "supplemental_balanced_stable": "Sensitivity: supplemental balanced allocation-stable cells"}[args.view]
    if args.paired_only:
        view_label += "; complete selected-arm pairs only"
    fig.text(0.5, 0.905, view_label, ha="center", fontsize=10, color="#424c55")
    config = report.get("args", {})
    matching_cells = [cell for cell in source_cells
                      if cell.get("actual_turns", cell.get("max_turns")) == selected_turns
                      and cell.get("evolution_policy", "upstream") == args.policy]
    counts = source_counts(matching_cells)
    fig.text(0.5, 0.205,
             f"Source cells: planned {counts['planned']}; completed {counts['completed']}; failed {counts['failed']}; pending/running {counts['pending_or_running']}.",
             ha="center", fontsize=8.2, color="#424c55")
    penalty = config.get("repetition_penalty")
    fig.text(0.5, 0.155,
             f"Common decoding: temperature 0; seed {config.get('seed', 42)}; max_tokens {config.get('max_tokens', 'recorded')}; repetition_penalty {penalty if penalty is not None else 'omitted'}.",
             ha="center", fontsize=8.2, color="#424c55")
    tag_budget = config.get("tag_budget")
    tag_policy = config.get("tag_budget_policy") or "error"
    if tag_budget is None:
        tag_caption = f"Tag instruction: omitted; acceptance policy: {tag_policy} (default)."
    else:
        acceptance = "overshoots retained" if tag_policy == "observe" else "overshoots rejected"
        tag_caption = f"Tag instruction: at most {tag_budget}/list; acceptance policy: {tag_policy} ({acceptance})."
    if json_guidance:
        tag_caption += " Fixed JSON schema on fused/joint only."
    fig.text(0.5, 0.105, tag_caption, ha="center", fontsize=8.2, color="#424c55")
    replay_note = "Completed cold-arm build+QA outputs can precede another arm's failure; replay verification is separate."
    fig.text(0.5, 0.07, replay_note, ha="center", fontsize=8.2, color="#424c55")
    fig.text(0.5, 0.03,
             ("Bars: paired arithmetic means; dots: repeats. Labels: paired geometric build speedup / mean ΔF1. Shared GPU; no confidence interval."
              if args.paired_only else "Bars: available within-conversation repeat means. Dots: same selected QA per repeat. Shared GPU; no confidence interval implied."),
             ha="center", fontsize=8.2, color="#424c55")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.36, wspace=0.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    if args.output.suffix.lower() == ".svg":
        # Matplotlib paths contain cosmetic trailing blanks; keep source diffs clean.
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    if args.png and args.output.suffix.lower() != ".png":
        fig.savefig(args.output.with_suffix(".png"), dpi=180, facecolor="white")
    plt.close(fig)
    data_path = args.output.with_suffix(".data.json")
    data_path.write_text(json.dumps({"source_suite": str(args.suite), "view": args.view,
                                     "confirmation_suite": str(args.confirmation_suite) if args.confirmation_suite else None,
                                     "source_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                                     "analysis_script": str(Path(__file__).resolve()),
                                     "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "analysis_dependencies_sha256": {"scripts/summarize_suite.py": hashlib.sha256(Path(__file__).with_name("summarize_suite.py").read_bytes()).hexdigest()},
                                     "source_cell_counts": counts, "plotted_cells_with_complete_arms": pair_count,
                                     "completed_arm_counts": arm_counts, "replay_note": replay_note,
                                     "paired_only": args.paired_only, "paired_annotations": pair_annotations,
                                     "tag_budget": tag_budget, "tag_budget_policy": tag_policy,
                                     "structured_policy_outputs": json_guidance,
                                     "arms": [arm for arm, _, _ in arms], "arm_offsets": offsets,
                                     "bar_slot_width": width,
                                     "matplotlib_version": matplotlib.__version__,
                                     "rows": rows, "metric_definitions": panels}, indent=2) + "\n")
    print(args.output)
    print(data_path)


if __name__ == "__main__":
    main()
