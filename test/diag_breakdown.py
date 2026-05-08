"""Diagnose the gap between _compute_peak_breakdown sum and avg_cumulative_mem peak.

At peak_idx, sum(avg_output_sizes of alive nodes) should approximately equal
avg_cumulative_mem[peak_idx] (CUDA-real bytes). Any gap is either:
  - real untracked bytes (workspace, allocator overhead), or
  - a bug in how we walk the alive set.

Usage: PYTHONPATH=. python test/diag_breakdown.py Resnet18 64
"""

import sys
import torch

from benchmarks import Experiment
from graph_prof import GraphProfiler, NodeType
from graph_tracer import compile


def main():
    model_name = sys.argv[1]
    bs = int(sys.argv[2])
    print(f"[{model_name} bs={bs}]")

    torch.cuda.empty_cache()
    exp = Experiment(model_name, batch_size=bs)
    exp.init_opt_states()

    captured = {}

    def xform(gm, args):
        prof = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(2):
                prof.run(*args)
            prof.reset_stats()
            for _ in range(3):
                prof.run(*args)
        prof.aggregate_stats()
        captured["prof"] = prof
        return gm

    compiled = compile(exp.train_step, xform)
    compiled(exp.model, exp.optimizer, exp.example_inputs)
    prof = captured["prof"]

    peak_idx = prof.avg_cumulative_mem.index(max(prof.avg_cumulative_mem))
    cuda_peak = prof.avg_cumulative_mem[peak_idx]
    print(f"  peak_idx={peak_idx}  cuda_peak={cuda_peak/1e6:.1f} MB")

    # Walk alive nodes at peak_idx (same logic as _compute_peak_breakdown)
    alive = []
    for i, node in enumerate(prof.node_order):
        if i > peak_idx:
            break
        if node.op == "output":
            continue
        last_user_idx = -1
        for u in node.users:
            uid = prof.order_index.get(u, -1)
            if uid > last_user_idx:
                last_user_idx = uid
        if last_user_idx >= peak_idx or (last_user_idx == -1 and i <= peak_idx):
            size = prof.avg_output_sizes.get(node, 0.0)
            if size > 0:
                nt = prof.node_type.get(node, NodeType.OTHER)
                alive.append((node.name, nt, size))

    breakdown_total = sum(s for _, _, s in alive)
    by_type = {}
    for _, nt, s in alive:
        by_type[nt] = by_type.get(nt, 0.0) + s
    for nt, total in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {nt.name:>10s}: {total/1e6:>7.1f} MB")
    print(f"  alive sum = {breakdown_total/1e6:.1f} MB")
    print(f"  gap (cuda_peak - alive_sum) = {(cuda_peak - breakdown_total)/1e6:.1f} MB")

    # Top 10 largest alive tensors
    print("\n  top-10 largest alive tensors at peak:")
    for name, nt, s in sorted(alive, key=lambda x: -x[2])[:10]:
        print(f"    {name:>40s}  {nt.name:>10s}  {s/1e6:>7.2f} MB")

    # Is the gap from running through optimizer too? Check what's at peak_idx.
    print(f"\n  node at peak_idx: {prof.node_order[peak_idx].name}")
    print(f"  node at peak_idx-1: {prof.node_order[max(0, peak_idx-1)].name}")
    print(f"  node at peak_idx+1: {prof.node_order[min(len(prof.node_order)-1, peak_idx+1)].name}")

    # Are all params/grads/opt_state classified correctly?
    counts_by_type = {nt: 0 for nt in NodeType}
    for n in prof.node_order:
        nt = prof.node_type.get(n, NodeType.OTHER)
        counts_by_type[nt] += 1
    print(f"\n  total node counts by type: { {nt.name: c for nt, c in counts_by_type.items()} }")


if __name__ == "__main__":
    main()
