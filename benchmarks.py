import importlib
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torchvision.models import resnet18, resnet50, resnet152
from graph_prof import GraphProfiler, NodeType
from graph_tracer import SEPFunction, compile

model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
    "Resnet152",
]

model_batch_sizes: Dict[str, int] = {
    "Transformer": 512,
    "Resnet18": 64,
    "Resnet50": 64,
    "Resnet152": 64,
}


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        assert (
            model_name in model_names
        ), f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        # μ-TWO opt-in scheduling: leave both as None for baseline mode.
        # Set exactly one before invoking the compiled train_step.
        self.mu_two_budget_bytes: int | None = None
        self.mu_two_budget_fraction: float | None = None
        # When True, suppress baseline + rewritten plot/CSV writes from
        # graph_transformation so a sweep harness doesn't clobber the Phase 1
        # deliverables. Profiling itself still runs.
        self.quiet_baseline_plots: bool = False
        # Populated by graph_transformation when a μ-TWO pass runs.
        self.baseline_peak_bytes: int | None = None
        self.predicted_peak_bytes: int | None = None
        self.measured_peak_bytes: int | None = None
        self.reached_budget: bool | None = None
        self.recomp_picks: Dict[fx.Node, Any] | None = None
        self.graph_profiler_after: GraphProfiler | None = None
        # Snapshot of baseline _compute_peak_breakdown() captured BEFORE the
        # rewriter mutates user lists. Calling it after rewrite would walk
        # rewired user sets and report stale "dead at peak" for activations
        # whose backward edge moved to a clone.
        self.baseline_breakdown_bytes: Dict[Any, float] | None = None

        if self.model_name == "Transformer":

            vocab_size = 2048
            bsz, seq_len = self.batch_size, 256
            with torch.device(dev):
                model_args = ModelArgs(
                    n_layers=8,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                self.model = Transformer(model_args)
            src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (src, tgt)

            def transformer_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = transformer_train_step
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2, fused=True, capturable=True
            )

        elif self.model_name in ["Resnet18", "Resnet50", "Resnet152"]:
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                if self.model_name == "Resnet18":
                    self.model = resnet18()
                elif self.model_name == "Resnet50":
                    self.model = resnet50()
                else:
                    self.model = resnet152()

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-2, fused=True, capturable=True
            )
            self.train_step = resnet_train_step

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        # print(gm.graph.print_tabular())
        out_dir = f"output/{self.model_name}"
        os.makedirs(out_dir, exist_ok=True)
        warm_up_iters, profile_iters = 2, 3
        self.graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                self.graph_profiler.run(*args)
            self.graph_profiler.reset_stats()

            for _ in range(profile_iters):
                self.graph_profiler.run(*args)
            self.graph_profiler.aggregate_stats()
            if not self.quiet_baseline_plots:
                self.graph_profiler.print_stats(
                    f"{out_dir}/node_stats.csv",
                    f"{out_dir}/activation_lifecycle.csv",
                )
                self.graph_profiler.plot_memory_timeline(
                    f"{out_dir}/memory_timeline.png"
                )
                self.graph_profiler.plot_memory_breakdown(
                    f"{out_dir}/memory_breakdown.png"
                )
                self.graph_profiler.plot_memory_stacked_timeline(
                    f"{out_dir}/memory_stacked_timeline.png"
                )

        if (
            self.mu_two_budget_bytes is not None
            or self.mu_two_budget_fraction is not None
        ):
            from mu_two_core import simulate
            from mu_two_scheduler import greedy_recompute
            from mu_two_rewrite import rewrite_recomputes

            self.baseline_breakdown_bytes = (
                self.graph_profiler._compute_peak_breakdown()
            )

            baseline_peak = max(simulate(self.graph_profiler, recomps={}))
            self.baseline_peak_bytes = int(baseline_peak)
            if self.mu_two_budget_bytes is not None:
                budget = int(self.mu_two_budget_bytes)
            else:
                budget = int(baseline_peak * self.mu_two_budget_fraction)

            recomps, reached = greedy_recompute(
                self.graph_profiler,
                budget=budget,
            )
            self.recomp_picks = recomps
            self.reached_budget = reached
            self.predicted_peak_bytes = int(max(simulate(self.graph_profiler, recomps)))

            gm = rewrite_recomputes(gm, self.graph_profiler, recomps)

            # Re-profile rewritten graph to measure realized peak. Same iter
            # counts as baseline so peaks are comparable. Misclassification of
            # cloned recompute nodes (they get tagged ACT) only affects the
            # breakdown plots, not avg_cumulative_mem which is sourced from
            # raw torch.cuda.memory_allocated() snapshots.
            torch.cuda.empty_cache()
            prof_after = GraphProfiler(gm)
            with torch.no_grad():
                for _ in range(warm_up_iters):
                    prof_after.run(*args)
                prof_after.reset_stats()
                for _ in range(profile_iters):
                    prof_after.run(*args)
                prof_after.aggregate_stats()
            self.graph_profiler_after = prof_after
            self.measured_peak_bytes = int(max(prof_after.avg_cumulative_mem))

            if not self.quiet_baseline_plots:
                prof_after.plot_memory_timeline(f"{out_dir}/memory_timeline_mu_two.png")
                prof_after.plot_memory_breakdown(
                    f"{out_dir}/memory_breakdown_mu_two.png"
                )
                prof_after.plot_memory_stacked_timeline(
                    f"{out_dir}/memory_stacked_timeline_mu_two.png"
                )

            compare_dir = f"{out_dir}/timeline_compare"
            os.makedirs(compare_dir, exist_ok=True)
            _plot_memory_timeline_overlay(
                self.graph_profiler,
                prof_after,
                save_path=f"{compare_dir}/bs{self.batch_size}.png",
                title=(
                    f"Memory Timeline: Baseline vs μ-TWO ({self.model_name}, "
                    f"bs={self.batch_size})"
                ),
            )

        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


experiment_batch_sizes: Dict[str, List[int]] = {
    # Sweep includes small bs (where greedy can't reach the budget) so the
    # grouped bar chart shows the regime where μ-TWO has no leverage. The
    # graph_transformation bail-out handles the small-bs case by returning
    # the baseline graph unchanged when greedy exhausts candidates.
    "Resnet18": [4, 8, 16, 32, 64],
    "Resnet50": [2, 4, 8, 16, 32, 64],
    "Resnet152": [4, 8, 16, 32],
    "Transformer": [4, 16, 64, 256, 512],
}


def run_batch_size_experiment(
    model_name: str,
    batch_sizes: List[int] | None = None,
    save_path: str | None = None,
) -> Dict[int, float]:
    """Run profiling at multiple batch sizes and plot peak memory vs batch size.

    Args:
        model_name: One of the model_names list entries.
        batch_sizes: List of batch sizes to test. Defaults to experiment_batch_sizes.
        save_path: File path for the bar chart. Defaults to '{model_name}_peak_memory.png'.

    Returns:
        Dict mapping batch_size -> peak_memory_mb.
    """
    if batch_sizes is None:
        batch_sizes = experiment_batch_sizes.get(model_name, [4, 8, 16])
    if save_path is None:
        os.makedirs(f"output/{model_name}", exist_ok=True)
        save_path = f"output/{model_name}/peak_memory.png"

    results: Dict[int, float] = {}

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Profiling {model_name} with batch_size={bs}")
        print(f"{'='*60}")

        try:
            torch.cuda.empty_cache()
            exp = Experiment(model_name, bs)
            exp.init_opt_states()
            # compile() caches on the wrapper function, so we get a fresh
            # compiled_fn per call (new wrapper = new cache slot)
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

            peak_mb = exp.graph_profiler.get_peak_memory_mb()
            results[bs] = peak_mb
            print(f"  -> Peak memory: {peak_mb:.2f} MB")

            # Clean up to free GPU memory
            del exp, compiled_fn
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  -> OOM at batch_size={bs}, skipping.")
                torch.cuda.empty_cache()
            else:
                raise

    # Generate bar chart
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        batch_sizes_run = sorted(results.keys())
        peak_mems = [results[bs] for bs in batch_sizes_run]

        bars = ax.bar(
            [str(bs) for bs in batch_sizes_run],
            peak_mems,
            color="steelblue",
            edgecolor="black",
        )

        # Add value labels on bars
        for bar, val in zip(bars, peak_mems):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(peak_mems) * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Mini-Batch Size")
        ax.set_ylabel("Peak GPU Memory (MB)")
        ax.set_title(f"Peak Memory vs Mini-Batch Size ({model_name})")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nBar chart saved to {save_path}")
        plt.close(fig)

    return results


def _plot_memory_timeline_overlay(
    prof_baseline: GraphProfiler,
    prof_after: GraphProfiler,
    save_path: str,
    title: str,
) -> None:
    """Overlay baseline and μ-TWO memory-over-execution curves.

    Both curves come from `avg_cumulative_mem` (real torch.cuda.memory_allocated
    snapshots), so they're directly comparable. The two graphs have different
    node counts (μ-TWO adds clones), so we plot each over its own normalized
    [0, 1] execution-progress x-axis instead of raw node index — this lets the
    fwd→bwd "mountain" line up visually.
    """
    if not prof_baseline.avg_cumulative_mem or not prof_after.avg_cumulative_mem:
        print(f"  skip overlay (missing profile data) → {save_path}")
        return

    mb = 1024 * 1024
    base = [m / mb for m in prof_baseline.avg_cumulative_mem]
    after = [m / mb for m in prof_after.avg_cumulative_mem]

    x_base = np.linspace(0, 1, len(base))
    x_after = np.linspace(0, 1, len(after))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x_base,
        base,
        color="steelblue",
        linewidth=1.2,
        label=f"Baseline (peak {max(base):.1f} MB)",
    )
    ax.plot(
        x_after,
        after,
        color="darkorange",
        linewidth=1.2,
        label=f"μ-TWO (peak {max(after):.1f} MB)",
    )
    ax.fill_between(x_base, base, alpha=0.15, color="steelblue")
    ax.fill_between(x_after, after, alpha=0.15, color="darkorange")

    ax.set_xlabel("Execution Progress (normalized)")
    ax.set_ylabel("GPU Memory Allocated (MB)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def _plot_grouped_bars(
    batch_sizes: List[int],
    baseline_vals: List[float],
    mu_two_vals: List[float],
    committed_flags: List[bool],
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """Two-series grouped bar chart: baseline vs μ-TWO at each batch size.

    `committed_flags[i]=False` (no picks committed — peak-aware filtered all
    candidates) is rendered with a hatched μ-TWO bar so it's obvious where
    the rewrite was a no-op vs. where it actually changed the graph.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.38

    bars_b = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline",
        color="steelblue",
        edgecolor="black",
    )
    bars_m = ax.bar(
        x + width / 2,
        mu_two_vals,
        width,
        label="μ-TWO",
        color="darkorange",
        edgecolor="black",
    )
    for bar, committed in zip(bars_m, committed_flags):
        if not committed:
            bar.set_hatch("//")

    all_vals = baseline_vals + mu_two_vals
    ymax = max(all_vals) if all_vals else 1.0
    for bars, vals in [(bars_b, baseline_vals), (bars_m, mu_two_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Mini-Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def _plot_grouped_stacked_bars(
    batch_sizes: List[int],
    baseline_breakdowns: List[Dict[NodeType, float]],
    mu_two_breakdowns: List[Dict[NodeType, float]],
    committed_flags: List[bool],
    title: str,
    save_path: str,
) -> None:
    """Grouped stacked bar chart: per bs, two bars (baseline / μ-TWO) each
    stacked by NodeType. ACT is the top segment so the rewrite's effect is
    immediately visible. Bar totals (peak in MB) are labeled above each bar.

    `committed_flags[i]=False` (no picks committed — rewrite was a no-op)
    is rendered with a hatched μ-TWO bar.

    Note: cloned recompute nodes in the rewritten gm get classified as ACT by
    the post-rewrite GraphProfiler, so the μ-TWO ACT segment includes them —
    that's the right visual story (ACT shrinks net of the recompute clones).
    """
    # Bottom-to-top stack order. ACT goes last so it sits on top.
    stack_order = [
        NodeType.PARAM,
        NodeType.GRAD,
        NodeType.OPT_STATE,
        NodeType.OTHER,
        NodeType.ACT,
    ]
    colors = {
        NodeType.PARAM: "#3498db",
        NodeType.GRAD: "#2ecc71",
        NodeType.OPT_STATE: "#f39c12",
        NodeType.OTHER: "#95a5a6",
        NodeType.ACT: "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(batch_sizes))
    width = 0.38

    base_bottom = np.zeros(len(batch_sizes))
    mu_bottom = np.zeros(len(batch_sizes))
    handles = {}

    for nt in stack_order:
        base_seg = np.array([bd.get(nt, 0.0) for bd in baseline_breakdowns])
        mu_seg = np.array([bd.get(nt, 0.0) for bd in mu_two_breakdowns])
        ax.bar(
            x - width / 2,
            base_seg,
            width,
            bottom=base_bottom,
            color=colors[nt],
            edgecolor="black",
            linewidth=0.4,
        )
        rects_m = ax.bar(
            x + width / 2,
            mu_seg,
            width,
            bottom=mu_bottom,
            color=colors[nt],
            edgecolor="black",
            linewidth=0.4,
        )
        handles[nt] = rects_m
        base_bottom += base_seg
        mu_bottom += mu_seg

    # Hatch the entire μ-TWO bar (every segment) when no picks were committed.
    for i, committed in enumerate(committed_flags):
        if not committed:
            for nt in stack_order:
                handles[nt][i].set_hatch("//")

    # Label each bar with its total
    ymax = max(
        base_bottom.max() if len(base_bottom) else 0,
        mu_bottom.max() if len(mu_bottom) else 0,
        1.0,
    )
    for xi, total in zip(x - width / 2, base_bottom):
        ax.text(
            xi,
            total + ymax * 0.01,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for xi, total in zip(x + width / 2, mu_bottom):
        ax.text(
            xi,
            total + ymax * 0.01,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # x-axis: batch sizes with "B" / "μ" sub-labels handled by group spacing
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Mini-Batch Size  (left bar = Baseline, right bar = μ-TWO)")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(title)

    # Legend: one swatch per NodeType, ACT first so it's prominent
    legend_order = [
        NodeType.ACT,
        NodeType.OPT_STATE,
        NodeType.GRAD,
        NodeType.PARAM,
        NodeType.OTHER,
    ]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[nt], ec="black", lw=0.4)
        for nt in legend_order
    ]
    legend_labels = [nt.name for nt in legend_order]
    ax.legend(legend_handles, legend_labels, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def run_baseline_vs_mu_two_sweep(
    model_name: str,
    budget_fraction: float = 0.7,
    batch_sizes: List[int] | None = None,
    save_dir: str = "output",
) -> Dict[int, Dict[str, Any]]:
    """Sweep batch sizes; for each, run one Experiment with μ-TWO enabled and
    capture both baseline and rewritten peak / latency. Emit two grouped bar
    charts (peak memory, iteration latency).

    Returns: {bs: {baseline_peak_mb, mu_two_peak_mb,
                   baseline_latency_ms, mu_two_latency_ms,
                   reached_budget}}.
    """
    if batch_sizes is None:
        batch_sizes = experiment_batch_sizes.get(model_name, [4, 8, 16])
    os.makedirs(f"{save_dir}/{model_name}", exist_ok=True)

    results: Dict[int, Dict[str, Any]] = {}

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"mu-TWO sweep: {model_name} bs={bs} budget={budget_fraction}")
        print(f"{'='*60}")
        try:
            torch.cuda.empty_cache()
            exp = Experiment(model_name, bs)
            exp.init_opt_states()
            # When the sweep emits a single bs there's no clobber risk and
            # the per-model timeline/stacked/breakdown plots are useful, so
            # leave quiet_baseline_plots False.
            exp.quiet_baseline_plots = len(batch_sizes) > 1
            exp.mu_two_budget_fraction = budget_fraction
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

            # Both peaks pulled from each profiler's avg_cumulative_mem
            # (real torch.cuda.memory_allocated snapshots) so they're
            # apples-to-apples. Using exp.baseline_peak_bytes here would
            # mix simulator output against CUDA-real and produce nonsense
            # bars at bail-out (where the gm is identical in both runs).
            baseline_peak_mb = exp.graph_profiler.get_peak_memory_mb()
            mu_two_peak_mb = exp.graph_profiler_after.get_peak_memory_mb()
            baseline_lat_ms = sum(exp.graph_profiler.avg_runtimes.values())
            mu_two_lat_ms = sum(exp.graph_profiler_after.avg_runtimes.values())

            results[bs] = {
                "baseline_peak_mb": baseline_peak_mb,
                "mu_two_peak_mb": mu_two_peak_mb,
                "baseline_latency_ms": baseline_lat_ms,
                "mu_two_latency_ms": mu_two_lat_ms,
                "reached_budget": bool(exp.reached_budget),
                "picks": len(exp.recomp_picks or {}),
                # Bytes-at-peak by NodeType, in MB. Sums to ~CUDA peak.
                # Baseline uses the pre-rewrite snapshot stored on exp;
                # rewriter mutates user lists so calling _compute_peak_breakdown
                # late on graph_profiler would undercount alive activations.
                "baseline_breakdown_mb": {
                    nt: b / (1024 * 1024)
                    for nt, b in (exp.baseline_breakdown_bytes or {}).items()
                },
                "mu_two_breakdown_mb": {
                    nt: b / (1024 * 1024)
                    for nt, b in exp.graph_profiler_after._compute_peak_breakdown().items()
                },
            }
            picks = len(exp.recomp_picks or {})
            if picks == 0:
                flag = " (no picks committed)"
            elif not exp.reached_budget:
                flag = f" ({picks} picks, budget not reached)"
            else:
                flag = f" ({picks} picks)"
            print(
                f"  peak: {baseline_peak_mb:.1f} -> {mu_two_peak_mb:.1f} MB"
                f" | latency: {baseline_lat_ms:.2f} -> {mu_two_lat_ms:.2f} ms"
                f"{flag}"
            )

            del exp, compiled_fn
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  -> OOM at bs={bs}, skipping.")
                torch.cuda.empty_cache()
            else:
                raise

    if not results:
        return results

    bs_run = sorted(results.keys())
    baseline_peaks = [results[bs]["baseline_peak_mb"] for bs in bs_run]
    mu_two_peaks = [results[bs]["mu_two_peak_mb"] for bs in bs_run]
    baseline_lats = [results[bs]["baseline_latency_ms"] for bs in bs_run]
    mu_two_lats = [results[bs]["mu_two_latency_ms"] for bs in bs_run]
    # Hatch bars when no picks were committed (rewrite was a no-op). Reaching
    # or missing budget is no longer the bail criterion — we always commit
    # whatever greedy returned, so the visual signal is "did anything change?"
    committed = [results[bs]["picks"] > 0 for bs in bs_run]

    _plot_grouped_bars(
        bs_run,
        baseline_peaks,
        mu_two_peaks,
        committed,
        ylabel="Peak GPU Memory (MB)",
        title=(
            f"Peak Memory vs Mini-Batch Size ({model_name}, "
            f"μ-TWO budget={budget_fraction:g}× baseline)"
        ),
        save_path=f"{save_dir}/{model_name}/peak_memory_baseline_vs_mu_two.png",
    )
    _plot_grouped_bars(
        bs_run,
        baseline_lats,
        mu_two_lats,
        committed,
        ylabel="Iteration Latency (ms)",
        title=(
            f"Iteration Latency vs Mini-Batch Size ({model_name}, "
            f"μ-TWO budget={budget_fraction:g}× baseline)"
        ),
        save_path=f"{save_dir}/{model_name}/latency_baseline_vs_mu_two.png",
    )
    _plot_grouped_stacked_bars(
        bs_run,
        [results[bs]["baseline_breakdown_mb"] for bs in bs_run],
        [results[bs]["mu_two_breakdown_mb"] for bs in bs_run],
        committed,
        title=(
            f"Peak Memory Breakdown vs Mini-Batch Size ({model_name}, "
            f"μ-TWO budget={budget_fraction:g}× baseline)"
        ),
        save_path=f"{save_dir}/{model_name}/peak_memory_breakdown_baseline_vs_mu_two.png",
    )

    return results


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--experiment":
        models_to_run = [args[1]] if len(args) > 1 else model_names
        for model_name in models_to_run:
            if model_name in experiment_batch_sizes:
                print(f"\n{'#'*60}")
                print(f"# Running batch size experiment for {model_name}")
                print(f"{'#'*60}")
                results = run_batch_size_experiment(model_name)
                print(f"\nResults for {model_name}: {results}")
    elif args and args[0] == "--mu-two-sweep":
        model = args[1] if len(args) > 1 else "Transformer"
        fraction = float(args[2]) if len(args) > 2 else 0.7
        print(f"\n{'#'*60}")
        print(f"# mu-TWO sweep: {model} @ budget={fraction}")
        print(f"{'#'*60}")
        run_baseline_vs_mu_two_sweep(model, budget_fraction=fraction)
    else:
        models_to_run = [args[0]] if args else model_names
        for name in models_to_run:
            print(f"\n{'#'*60}")
            print(f"# Profiling {name} (batch_size={model_batch_sizes[name]})")
            print(f"{'#'*60}")
            torch.cuda.empty_cache()
            exp = Experiment(name, model_batch_sizes[name])
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
            # Clear memory for next iteration
            del exp, compiled_fn
            torch.cuda.empty_cache()
