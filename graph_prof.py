import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OPT_STATE = 3
    OTHER = 4


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # --- Profiling accumulators ---
        self.node_runtimes: Dict[fx.Node, List[float]] = defaultdict(list)
        self.node_mem_deltas: Dict[fx.Node, List[int]] = defaultdict(list)
        self.node_output_sizes: Dict[fx.Node, List[int]] = defaultdict(list)
        self.cumulative_mem: List[List[int]] = []
        self._current_iter_mem: List[int] = []

        # Aggregated stats (populated by aggregate_stats)
        self.avg_runtimes: Dict[fx.Node, float] = {}
        self.avg_mem_deltas: Dict[fx.Node, float] = {}
        self.avg_output_sizes: Dict[fx.Node, float] = {}
        self.avg_cumulative_mem: List[float] = []

        # --- Static analysis ---
        self.node_order: List[fx.Node] = []
        self.order_index: Dict[fx.Node, int] = {}
        self.sep_node: Optional[fx.Node] = None
        self.sep_backward_node: Optional[fx.Node] = None
        self.optimizer_node: Optional[fx.Node] = None

        self.node_region: Dict[fx.Node, int] = {}  # 0=fwd, 1=loss, 2=bwd, 3=opt
        self.node_type: Dict[fx.Node, NodeType] = {}
        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()
        self.opt_state_nodes: Set[fx.Node] = set()
        self.activation_nodes: Set[fx.Node] = set()
        self.last_forward_access: Dict[fx.Node, fx.Node] = {}
        self.first_backward_access: Dict[fx.Node, fx.Node] = {}

        # 1. Build node_order, order_index, and locate sentinel nodes
        found_sep_backward = False
        for idx, node in enumerate(self.module.graph.nodes):
            self.node_order.append(node)
            self.order_index[node] = idx

            if node.op == OP.CALL_FUNCTION:
                if node.target == torch.ops.separator.sep.default:
                    self.sep_node = node
                elif node.target == torch.ops.separator.sep_backward.default:
                    self.sep_backward_node = node
                    found_sep_backward = True
                elif node.target == torch.ops.aten._fused_adam.default:
                    self.optimizer_node = node
                elif (
                    found_sep_backward
                    and self.optimizer_node is None
                    and "_foreach" in str(node.target)
                ):
                    # foreach-based optimizer: first _foreach op marks optimizer start
                    self.optimizer_node = node

        # 2. Assign regions
        region = 0  # start in FORWARD
        for node in self.node_order:
            self.node_region[node] = region
            # Transition after sentinel nodes
            if node is self.sep_node:
                region = 1  # LOSS
            elif node is self.sep_backward_node:
                region = 2  # BACKWARD
            elif node is self.optimizer_node:
                region = 3  # OPTIMIZER

        # 3. Identify params, grads, and optimizer states
        is_fused = (
            self.optimizer_node is not None
            and self.optimizer_node.target == torch.ops.aten._fused_adam.default
        )
        if is_fused:
            # _fused_adam args[0] = list of param nodes, args[1] = list of grad nodes
            param_list = self.optimizer_node.args[0]
            grad_list = self.optimizer_node.args[1]
            self.param_nodes = set(param_list) if param_list else set()
            self.grad_nodes = set(grad_list) if grad_list else set()
        else:
            # foreach-based optimizer or no optimizer node
            for node in self.node_order:
                if node.op == OP.PLACEHOLDER:
                    user_regions = {self.node_region.get(u, -1) for u in node.users}
                    # Params are used in fwd (region 0) and in opt (region 3)
                    if 0 in user_regions and 3 in user_regions:
                        self.param_nodes.add(node)

            # Grads: backward-region nodes that feed into optimizer-region nodes
            for node in self.node_order:
                if self.node_region[node] == 2:
                    for user in node.users:
                        if self.node_region.get(user, -1) == 3:
                            self.grad_nodes.add(node)
                            break

        # Optimizer states: placeholder nodes used ONLY in the optimizer region
        # (e.g., Adam's exp_avg, exp_avg_sq, step tensors)
        for node in self.node_order:
            if node.op == OP.PLACEHOLDER and node not in self.param_nodes:
                user_regions = {self.node_region.get(u, -1) for u in node.users}
                # Only used in optimizer region (region 3), or has no users
                if user_regions and user_regions <= {3}:
                    self.opt_state_nodes.add(node)

        # 4. Classify every node
        for node in self.node_order:
            if node in self.param_nodes:
                self.node_type[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_type[node] = NodeType.GRAD
            elif node in self.opt_state_nodes:
                self.node_type[node] = NodeType.OPT_STATE
            elif (
                self.node_region[node] == 2
                and node.op == OP.CALL_FUNCTION
            ):
                # All call_function nodes in the backward region are computing
                # gradients (intermediate or final). Tag them as GRAD.
                self.node_type[node] = NodeType.GRAD
                self.grad_nodes.add(node)
            elif (
                self.node_region[node] in (0, 1)
                and node.op != OP.PLACEHOLDER
                and any(self.node_region.get(u, -1) == 2 for u in node.users)
            ):
                self.node_type[node] = NodeType.ACT
                self.activation_nodes.add(node)
            else:
                self.node_type[node] = NodeType.OTHER

        # 5. Track activation lifecycle
        for act_node in self.activation_nodes:
            # Last forward access: among users in regions 0/1, highest order_index
            fwd_users = [
                u for u in act_node.users
                if self.node_region.get(u, -1) in (0, 1)
            ]
            if fwd_users:
                self.last_forward_access[act_node] = max(
                    fwd_users, key=lambda u: self.order_index[u]
                )

            # First backward access: among users in region 2, lowest order_index
            bwd_users = [
                u for u in act_node.users
                if self.node_region.get(u, -1) == 2
            ]
            if bwd_users:
                self.first_backward_access[act_node] = min(
                    bwd_users, key=lambda u: self.order_index[u]
                )

        # GPU availability warning (issued once at init)
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available. GraphProfiler will record zero-valued "
                "timing and memory data. Results will NOT be meaningful. "
                "Run on a CUDA-enabled GPU for accurate profiling.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    #  Activation recomputation cost estimation (static)
    # ------------------------------------------------------------------ #

    def _estimate_recomputation_cost(
        self, act_node: fx.Node
    ) -> Tuple[int, float]:
        """Estimate the cost to recompute an activation from its inputs.

        Walks backward from *act_node* through its forward-region inputs,
        counting the ops and summing their average runtimes until we reach
        either a placeholder (parameter / input) or another activation that
        would still be in memory (i.e., another node in activation_nodes).

        Returns (num_ops, total_time_ms).
        """
        visited: Set[fx.Node] = set()
        stack = [act_node]
        num_ops = 0
        total_time = 0.0

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            if node is not act_node:
                # Stop expanding at placeholders or other activations
                if node.op == OP.PLACEHOLDER:
                    continue
                if node in self.activation_nodes:
                    continue
                num_ops += 1
                total_time += self.avg_runtimes.get(node, 0.0)

            # Expand into this node's forward-region inputs
            for inp in node.all_input_nodes:
                if self.node_region.get(inp, -1) in (0, 1):
                    stack.append(inp)

        return num_ops, total_time

    # ------------------------------------------------------------------ #
    #  Execution
    # ------------------------------------------------------------------ #

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self._current_iter_mem = []
        result = super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )
        self.cumulative_mem.append(self._current_iter_mem)
        return result

    def _tensor_size_bytes(self, result: Any) -> int:
        """Return the byte size of a tensor result, handling tuples/lists."""
        if isinstance(result, torch.Tensor):
            return result.nelement() * result.element_size()
        if isinstance(result, (tuple, list)):
            return sum(self._tensor_size_bytes(r) for r in result)
        return 0

    def run_node(self, n: fx.Node) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = super().run_node(n)
            end.record()

            torch.cuda.synchronize()

            self.node_runtimes[n].append(start.elapsed_time(end))  # ms
            mem_after = torch.cuda.memory_allocated()
            self.node_mem_deltas[n].append(mem_after - mem_before)  # bytes
            self._current_iter_mem.append(mem_after)  # timeline
            self.node_output_sizes[n].append(self._tensor_size_bytes(result))
        else:
            result = super().run_node(n)
            self.node_runtimes[n].append(0.0)
            self.node_mem_deltas[n].append(0)
            self._current_iter_mem.append(0)
            self.node_output_sizes[n].append(self._tensor_size_bytes(result))

        return result

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def reset_stats(self) -> None:
        """Clear profiling accumulators. Does NOT clear static analysis data."""
        self.node_runtimes.clear()
        self.node_mem_deltas.clear()
        self.node_output_sizes.clear()
        self.cumulative_mem.clear()
        self._current_iter_mem = []
        self.avg_runtimes.clear()
        self.avg_mem_deltas.clear()
        self.avg_output_sizes.clear()
        self.avg_cumulative_mem = []

    def aggregate_stats(self) -> None:
        """Average profiling data over all recorded iterations."""
        for node in self.node_order:
            if node in self.node_runtimes and self.node_runtimes[node]:
                self.avg_runtimes[node] = (
                    sum(self.node_runtimes[node]) / len(self.node_runtimes[node])
                )
            else:
                self.avg_runtimes[node] = 0.0

            if node in self.node_mem_deltas and self.node_mem_deltas[node]:
                self.avg_mem_deltas[node] = (
                    sum(self.node_mem_deltas[node]) / len(self.node_mem_deltas[node])
                )
            else:
                self.avg_mem_deltas[node] = 0.0

            if node in self.node_output_sizes and self.node_output_sizes[node]:
                self.avg_output_sizes[node] = (
                    sum(self.node_output_sizes[node]) / len(self.node_output_sizes[node])
                )
            else:
                self.avg_output_sizes[node] = 0.0

        # Average cumulative memory timelines element-wise
        if self.cumulative_mem:
            num_iters = len(self.cumulative_mem)
            timeline_len = len(self.cumulative_mem[0])
            self.avg_cumulative_mem = []
            for i in range(timeline_len):
                total = sum(
                    self.cumulative_mem[j][i]
                    for j in range(num_iters)
                    if i < len(self.cumulative_mem[j])
                )
                count = sum(
                    1
                    for j in range(num_iters)
                    if i < len(self.cumulative_mem[j])
                )
                self.avg_cumulative_mem.append(total / count)

    # ------------------------------------------------------------------ #
    #  Peak memory breakdown
    # ------------------------------------------------------------------ #

    def _compute_peak_breakdown(self) -> Dict[NodeType, float]:
        """Compute memory consumed by each NodeType at the peak memory point.

        Determines which tensors are *alive* at the peak index (produced at or
        before the peak, with at least one user at or after the peak) and sums
        their avg_output_sizes by category.  This correctly accounts for
        pre-allocated tensors like parameters and optimizer states, which have
        zero mem_delta but still occupy memory at peak time.
        """
        breakdown: Dict[NodeType, float] = {nt: 0.0 for nt in NodeType}
        if not self.avg_cumulative_mem:
            return breakdown

        peak_idx = self.avg_cumulative_mem.index(max(self.avg_cumulative_mem))

        for i, node in enumerate(self.node_order):
            if i > peak_idx:
                break  # produced after peak — not alive at peak

            # A node's output is alive at the peak if it has at least one user
            # whose order_index >= peak_idx (consumed at or after the peak).
            # Placeholder nodes (graph inputs) are always alive until their
            # last user, so we check that too.
            if node.op == OP.OUTPUT:
                continue

            last_user_idx = -1
            for user in node.users:
                uid = self.order_index.get(user, -1)
                if uid > last_user_idx:
                    last_user_idx = uid

            # If the node has no users, its output may still be alive (e.g.,
            # unused placeholders); conservatively include if produced before peak.
            if last_user_idx >= peak_idx or (last_user_idx == -1 and i <= peak_idx):
                size = self.avg_output_sizes.get(node, 0.0)
                if size > 0:
                    breakdown[self.node_type.get(node, NodeType.OTHER)] += size

        return breakdown

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #

    def print_stats(
        self,
        csv_path: str = "node_stats.csv",
        act_csv_path: str = "activation_lifecycle.csv",
    ) -> None:
        """Write per-node profiling table to CSV and print summary statistics."""
        import csv

        region_names = {0: "FWD", 1: "LOSS", 2: "BWD", 3: "OPT"}

        fwd_time = 0.0
        bwd_time = 0.0
        opt_time = 0.0

        # Print header
        print(f"\n{'='*120}")
        print(
            f"{'Node Name':<40} {'Op':<18} {'Type':<10} {'Region':<6} "
            f"{'Avg Time (ms)':>14} {'Avg Mem Delta (B)':>18} {'Output Size (B)':>16}"
        )
        print(f"{'='*120}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Node Name", "Op", "Type", "Region",
                "Avg Time (ms)", "Avg Mem Delta (B)", "Output Size (B)",
            ])
            for node in self.node_order:
                name = node.name
                op = str(node.op)
                ntype = self.node_type.get(node, NodeType.OTHER).name
                region = region_names.get(self.node_region.get(node, -1), "?")
                avg_t = self.avg_runtimes.get(node, 0.0)
                avg_m = self.avg_mem_deltas.get(node, 0.0)
                avg_sz = self.avg_output_sizes.get(node, 0.0)

                r = self.node_region.get(node, -1)
                if r == 0:
                    fwd_time += avg_t
                elif r == 1:
                    fwd_time += avg_t
                elif r == 2:
                    bwd_time += avg_t
                elif r == 3:
                    opt_time += avg_t

                print(
                    f"{name[:38]:<40} {op[:16]:<18} {ntype:<10} {region:<6} "
                    f"{avg_t:>14.4f} {avg_m:>18.0f} {avg_sz:>16.0f}"
                )
                writer.writerow([name, op, ntype, region, f"{avg_t:.4f}", f"{avg_m:.0f}", f"{avg_sz:.0f}"])

        print(f"Node stats written to {csv_path}")

        # Summary
        print(f"\n{'='*120}")
        print("SUMMARY")
        print(f"{'='*120}")
        print(f"  Forward + Loss time:  {fwd_time:.4f} ms")
        print(f"  Backward time:        {bwd_time:.4f} ms")
        print(f"  Optimizer time:       {opt_time:.4f} ms")
        print(f"  Total time:           {fwd_time + bwd_time + opt_time:.4f} ms")
        print()
        print(f"  Parameters:           {len(self.param_nodes)}")
        print(f"  Activations:          {len(self.activation_nodes)}")
        print(f"  Gradients:            {len(self.grad_nodes)}")
        print(f"  Optimizer states:     {len(self.opt_state_nodes)}")

        # --- Peak memory breakdown ---
        breakdown = self._compute_peak_breakdown()
        total_breakdown = sum(breakdown.values())
        if total_breakdown > 0:
            print(f"\n{'='*120}")
            print("PEAK MEMORY BREAKDOWN")
            print(f"{'='*120}")
            for nt in NodeType:
                val_mb = breakdown[nt] / (1024 * 1024)
                pct = breakdown[nt] / total_breakdown * 100
                print(f"  {nt.name:<14} {val_mb:>10.2f} MB  ({pct:>5.1f}%)")
            print(f"  {'TOTAL':<14} {total_breakdown / (1024**2):>10.2f} MB")

        # --- Activation lifecycle ---
        if self.activation_nodes:
            print(f"\n{'='*120}")
            print("ACTIVATION LIFECYCLE")
            print(
                f"{'Activation':<32} {'Size (B)':>10} {'Last Fwd Use':<22} "
                f"{'First Bwd Use':<22} {'Idle Gap':>9} {'Recomp Ops':>11} {'Recomp (ms)':>12}"
            )
            print(f"{'-'*120}")

            sorted_acts = sorted(
                self.activation_nodes, key=lambda n: self.order_index[n]
            )
            act_data = []
            for act in sorted_acts:
                last_fwd = self.last_forward_access.get(act)
                first_bwd = self.first_backward_access.get(act)
                idle_gap = -1
                if last_fwd is not None and first_bwd is not None:
                    idle_gap = (
                        self.order_index[first_bwd] - self.order_index[last_fwd]
                    )
                recomp_ops, recomp_time = self._estimate_recomputation_cost(act)
                act_size = self.avg_output_sizes.get(act, 0.0)
                act_data.append(
                    (act, last_fwd, first_bwd, idle_gap, recomp_ops, recomp_time, act_size)
                )

            act_data.sort(key=lambda x: x[3], reverse=True)

            with open(act_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Activation", "Size (B)", "Last Fwd Use", "First Bwd Use",
                    "Idle Gap", "Recomp Ops", "Recomp (ms)",
                ])
                for act, last_fwd, first_bwd, idle_gap, recomp_ops, recomp_time, act_size in act_data:
                    print(
                        f"{act.name[:30]:<32} {act_size:>10.0f} "
                        f"{(last_fwd.name[:20] if last_fwd else 'N/A'):<22} "
                        f"{(first_bwd.name[:20] if first_bwd else 'N/A'):<22} "
                        f"{idle_gap:>9} {recomp_ops:>11} {recomp_time:>12.4f}"
                    )
                    writer.writerow([
                        act.name,
                        f"{act_size:.0f}",
                        last_fwd.name if last_fwd else "N/A",
                        first_bwd.name if first_bwd else "N/A",
                        idle_gap,
                        recomp_ops,
                        f"{recomp_time:.4f}",
                    ])
            print(f"Activation lifecycle written to {act_csv_path}")

        # Peak memory
        if self.avg_cumulative_mem:
            peak_bytes = max(self.avg_cumulative_mem)
            peak_mb = peak_bytes / (1024 * 1024)
            print(
                f"\n  Peak GPU memory:      {peak_mb:.2f} MB ({peak_bytes:.0f} bytes)"
            )
            if total_breakdown > 0:
                act_pct = breakdown[NodeType.ACT] / total_breakdown * 100
                print(
                    f"  Activations account for {act_pct:.1f}% of peak memory"
                )

        print(f"{'='*120}\n")

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #

    def plot_memory_timeline(self, save_path: str = "memory_timeline.png") -> None:
        """Plot average cumulative memory usage over node execution."""
        if not self.avg_cumulative_mem:
            print("No memory data to plot. Run aggregate_stats() first.")
            return

        mem_mb = [m / (1024 * 1024) for m in self.avg_cumulative_mem]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(mem_mb)), mem_mb, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Node Execution Index")
        ax.set_ylabel("GPU Memory Allocated (MB)")
        ax.set_title("Memory Timeline During Training Step")

        # Draw vertical lines at sep and sep_backward boundaries
        # Use distinct styles to avoid overlap confusion
        if self.sep_node is not None:
            sep_idx = self.order_index[self.sep_node]
            ax.axvline(
                x=sep_idx, color="green", linestyle="--", linewidth=2,
                alpha=0.8, label="End of Forward",
            )
        if self.sep_backward_node is not None:
            sep_bwd_idx = self.order_index[self.sep_backward_node]
            ax.axvline(
                x=sep_bwd_idx, color="red", linestyle=":", linewidth=2,
                alpha=0.8, label="Start of Backward",
            )

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Memory timeline saved to {save_path}")
        plt.close(fig)

    def plot_memory_stacked_timeline(
        self, save_path: str = "memory_stacked_timeline.png"
    ) -> None:
        """Plot a stacked area chart of memory by tensor type over execution."""
        if not self.avg_output_sizes:
            print("No data to plot. Run aggregate_stats() first.")
            return

        n = len(self.node_order)
        colors = {
            NodeType.PARAM: "#3498db",
            NodeType.ACT: "#e74c3c",
            NodeType.GRAD: "#2ecc71",
            NodeType.OPT_STATE: "#f39c12",
            NodeType.OTHER: "#95a5a6",
        }

        # Precompute last user index for each node
        last_user = {}
        for i, node in enumerate(self.node_order):
            if node.op == OP.OUTPUT:
                continue
            lui = -1
            for u in node.users:
                uid = self.order_index.get(u, -1)
                if uid > lui:
                    lui = uid
            # Nodes with no users: alive until end
            last_user[i] = lui if lui >= 0 else n - 1

        # Build per-type memory timelines using a sweep approach
        # Track alive memory per type; at each step add new node, remove dead ones
        type_timelines: Dict[NodeType, List[float]] = {
            nt: [0.0] * n for nt in NodeType
        }

        # alive_nodes[i] = (node_type, size) for node produced at index i
        alive: List[Tuple[int, NodeType, float]] = []  # (last_user_idx, type, size)

        for i, node in enumerate(self.node_order):
            if node.op == OP.OUTPUT:
                # Just copy previous step
                if i > 0:
                    for nt in NodeType:
                        type_timelines[nt][i] = type_timelines[nt][i - 1]
                continue

            # Start with previous step's values
            if i > 0:
                for nt in NodeType:
                    type_timelines[nt][i] = type_timelines[nt][i - 1]

            # Add this node's output
            size = self.avg_output_sizes.get(node, 0.0)
            nt = self.node_type.get(node, NodeType.OTHER)
            if size > 0:
                type_timelines[nt][i] += size
                alive.append((last_user[i], nt, size))

            # Remove nodes whose last user was the previous step
            still_alive = []
            for lui, ant, asize in alive:
                if lui < i:
                    type_timelines[ant][i] -= asize
                else:
                    still_alive.append((lui, ant, asize))
            alive = still_alive

        # Convert to MB — order so ACT is on top
        mb = 1024 * 1024
        stack_order = [NodeType.OTHER, NodeType.ACT, NodeType.PARAM, NodeType.GRAD, NodeType.OPT_STATE]
        stacks = []
        labels = []
        stack_colors = []
        for nt in stack_order:
            vals = [v / mb for v in type_timelines[nt]]
            if max(vals) > 0:
                stacks.append(vals)
                labels.append(nt.name)
                stack_colors.append(colors.get(nt, "#95a5a6"))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.stackplot(range(n), *stacks, labels=labels, colors=stack_colors, alpha=0.85)
        ax.set_xlabel("Node Execution Index")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory by Tensor Type Over Execution")

        if self.sep_node is not None:
            ax.axvline(
                x=self.order_index[self.sep_node], color="green",
                linestyle="--", linewidth=2, alpha=0.8, label="End of Forward",
            )
        if self.sep_backward_node is not None:
            ax.axvline(
                x=self.order_index[self.sep_backward_node], color="red",
                linestyle=":", linewidth=2, alpha=0.8, label="Start of Backward",
            )

        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Stacked memory timeline saved to {save_path}")
        plt.close(fig)

    def plot_memory_breakdown(self, save_path: str = "memory_breakdown.png") -> None:
        """Plot a stacked bar chart showing peak memory breakdown by tensor type."""
        breakdown = self._compute_peak_breakdown()
        total = sum(breakdown.values())
        if total == 0:
            print("No memory breakdown data. Run aggregate_stats() first.")
            return

        # Filter out zero categories for cleaner chart
        labels = []
        sizes_mb = []
        colors = {
            NodeType.ACT: "#e74c3c",
            NodeType.PARAM: "#3498db",
            NodeType.GRAD: "#2ecc71",
            NodeType.OPT_STATE: "#f39c12",
            NodeType.OTHER: "#95a5a6",
        }
        bar_colors = []

        for nt in NodeType:
            if breakdown[nt] > 0:
                labels.append(nt.name)
                sizes_mb.append(breakdown[nt] / (1024 * 1024))
                bar_colors.append(colors.get(nt, "#95a5a6"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Stacked bar chart
        bottom = 0.0
        for label, size, color in zip(labels, sizes_mb, bar_colors):
            pct = size / (total / (1024 * 1024)) * 100
            ax1.bar(
                "Peak Memory", size, bottom=bottom, color=color,
                label=f"{label} ({size:.1f} MB, {pct:.1f}%)", edgecolor="white",
            )
            bottom += size

        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Peak Memory Breakdown by Tensor Type")
        ax1.legend(loc="upper left", fontsize=9)

        # Pie chart
        ax2.pie(
            sizes_mb, labels=labels, colors=bar_colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10},
        )
        ax2.set_title("Peak Memory Distribution")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Memory breakdown saved to {save_path}")
        plt.close(fig)

    def get_peak_memory_mb(self) -> float:
        """Return peak memory in MB from the averaged cumulative timeline."""
        if not self.avg_cumulative_mem:
            return 0.0
        return max(self.avg_cumulative_mem) / (1024 * 1024)
