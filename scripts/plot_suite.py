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
from pathlib import Path
import statistics

from summarize_suite import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="SVG, PDF, or PNG output")
    parser.add_argument("--png", action="store_true", help="Also write a PNG preview next to SVG/PDF")
    parser.add_argument("--analysis", type=Path, help="Optional analyze_suite_views.py JSON")
    parser.add_argument("--view", choices=("planned_all", "planned_allocation_stable", "supplemental_balanced_stable"), default="planned_all")
    parser.add_argument("--confirmation-suite", type=Path)
    parser.add_argument("--policy", choices=("upstream", "fused"), default="upstream", help="Experiment policy if suite contains multiple policy groups")
    parser.add_argument("--turns", type=int, help="Required if suite has more than one turn count")
    args = parser.parse_args()
    report = json.loads(args.suite.read_text())
    derived = summarize(report, args.suite.parent)
    all_rows = [{**row, "origin": "primary"} for row in derived["cells"]]
    if args.confirmation_suite:
        confirmation = json.loads(args.confirmation_suite.read_text())
        all_rows += [{**row, "origin": "confirmation"} for row in
                     summarize(confirmation, args.confirmation_suite.parent)["cells"]]
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
        parser.error("No completed cells to plot")
    if args.turns is None and len(turn_counts) != 1:
        parser.error("Specify --turns when plotting a suite with multiple turn counts")
    selected_turns = args.turns if args.turns is not None else turn_counts[0]
    rows = [row for row in all_rows if row["turns"] == selected_turns and row["experiment_policy"] == args.policy]
    if not rows:
        parser.error(f"No completed cells with {selected_turns} turns")
    samples = sorted({row["sample_index"] for row in rows})
    arms = [("baseline", "Configured robust", "#778899"),
            ("optimized", "Upstream pipeline" if args.policy == "upstream" else "Fused pipeline", "#0072B2"),
            ("fused", "Fused evolution", "#D55E00")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "svg.fonttype": "none", "axes.spines.top": False,
                                "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    panels = [("build_seconds", "Memory build time", "Seconds"),
              ("http_requests_build", "Live requests during build", "HTTP requests"),
              ("token_f1", "QA quality", "Normalized token F1")]
    width = 0.24
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for arm_index, (arm, label, color) in enumerate(arms):
            for sample_position, sample in enumerate(samples):
                selected = [row for row in rows if row["sample_index"] == sample and row["arm"] == arm]
                if not selected:
                    continue
                values = [row[metric] for row in selected]
                x = sample_position + (arm_index - 1) * width
                axis.bar(x, statistics.mean(values), width=width * 0.9, color=color,
                         alpha=0.8, label=label if sample_position == 0 else None, zorder=2)
                for repeat_index, value in enumerate(values):
                    spread = (repeat_index - (len(values) - 1) / 2) * min(0.045, width / (len(values) + 1))
                    axis.scatter(x + spread, value, s=16, color="#111111", edgecolors="white",
                                 linewidths=0.35, zorder=3)
        axis.set_title(title, fontweight="bold", pad=12)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(samples)), [f"Conversation {sample}" for sample in samples])
        axis.grid(axis="y", color="#dce2e7", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        if metric == "token_f1":
            axis.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=3, frameon=False)
    pair_count = len({(row["origin"], row["cell"]) for row in rows})
    model = report.get("args", {}).get("model", "model")
    fig.suptitle(f"LoCoMo {selected_turns}-turn subsets · {model} · {pair_count} completed paired run{'s' if pair_count != 1 else ''}",
                 fontsize=14, fontweight="bold", y=0.99)
    view_label = {"planned_all": "All planned completed pairs",
                  "planned_allocation_stable": "Sensitivity: allocation-stable planned pairs",
                  "supplemental_balanced_stable": "Sensitivity: supplemental balanced allocation-stable pairs"}[args.view]
    fig.text(0.5, 0.905, view_label, ha="center", fontsize=10, color="#424c55")
    config = report.get("args", {})
    fig.text(0.5, 0.085,
             f"Common decoding: temperature 0; seed {config.get('seed', 42)}; max_tokens {config.get('max_tokens', 'recorded')}; repetition_penalty {config.get('repetition_penalty', 'omitted')}",
             ha="center", fontsize=8.2, color="#424c55")
    fig.text(0.5, 0.035,
             "Bars: within-conversation repeat means. Dots: individual runs of the same selected QA. Shared GPU; no confidence interval implied.",
             ha="center", fontsize=8.2, color="#424c55")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.29, wspace=0.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    if args.png and args.output.suffix.lower() != ".png":
        fig.savefig(args.output.with_suffix(".png"), dpi=180, facecolor="white")
    plt.close(fig)
    data_path = args.output.with_suffix(".data.json")
    data_path.write_text(json.dumps({"source_suite": str(args.suite), "view": args.view,
                                     "confirmation_suite": str(args.confirmation_suite) if args.confirmation_suite else None,
                                     "source_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                                     "matplotlib_version": matplotlib.__version__,
                                     "rows": rows, "metric_definitions": panels}, indent=2) + "\n")
    print(args.output)
    print(data_path)


if __name__ == "__main__":
    main()
