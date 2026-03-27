import importlib
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
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
from graph_prof import GraphProfiler
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
        os.makedirs("output", exist_ok=True)
        warm_up_iters, profile_iters = 2, 3
        self.graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                self.graph_profiler.run(*args)
            self.graph_profiler.reset_stats()

            for _ in range(profile_iters):
                self.graph_profiler.run(*args)
            self.graph_profiler.aggregate_stats()
            self.graph_profiler.print_stats(
                f"output/{self.model_name}_node_stats.csv",
                f"output/{self.model_name}_activation_lifecycle.csv",
            )
            self.graph_profiler.plot_memory_timeline(
                f"output/{self.model_name}_memory_timeline.png"
            )
            self.graph_profiler.plot_memory_breakdown(
                f"output/{self.model_name}_memory_breakdown.png"
            )
            self.graph_profiler.plot_memory_stacked_timeline(
                f"output/{self.model_name}_memory_stacked_timeline.png"
            )

        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


experiment_batch_sizes: Dict[str, List[int]] = {
    "Resnet18": [4, 8, 16, 32, 64],
    "Resnet50": [2, 4, 8, 16],
    "Resnet152": [1, 2, 4, 8, 16, 32, 64],
    "Transformer": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
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
        os.makedirs("output", exist_ok=True)
        save_path = f"output/{model_name}_peak_memory.png"

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
