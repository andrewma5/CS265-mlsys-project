"""μ-TWO graph rewriter — Step 8 of TENTATIVE_PLAN.md.

Materializes the scheduler's recompute decisions into the joint FX graph:
for each picked activation A, clones a self-contained recompute subgraph
just before first_bwd_use(A) and rewires A's backward consumers to use the
clone. The original A loses its backward edge so it is freed at last_fwd_use.
"""

from typing import Dict, List

import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch._functorch._aot_autograd.descriptors import DummyAOTOutput

from activation_checkpoint import replace_subsequent_uses_of
from graph_prof import GraphProfiler
from mu_two_core import ActivationMeta


def rewrite_recomputes(
    gm: fx.GraphModule,
    prof: GraphProfiler,
    recomps: Dict[fx.Node, ActivationMeta],
) -> fx.GraphModule:
    """Insert recompute clones for every (act, meta) in *recomps*.

    Strategy per pick A:
      1. Extract the transitive forward subgraph from inputs=meta.recomp_srcs
         to outputs=[A]. Algorithms E + F (Steps 5–6) maintain the invariant
         that recomp_srcs contains only still-resident nodes (placeholders or
         not-picked activations), so the extraction terminates at a clean
         boundary. The build-time meta.recomp_subgraph is *not* used — it is
         stale post-cascade; the live FX graph is the source of truth.
      2. Clone the extracted interior nodes into gm.graph immediately before
         first_bwd_use(A). Other-picked activations that appear as interior
         ops (cascading dependencies) are cloned as fresh internal copies —
         each picked A gets its own self-contained recompute subgraph
         ("one clone per consumer", per TENTATIVE_PLAN §11).
      3. Rewire backward consumers of A to use the cloned A. Forward consumers
         (all sitting before first_bwd_use) are untouched, so the original A
         is still produced in the forward but loses its backward edge — freed
         at last_fwd_use, exactly what the simulator predicted.

    Picks are processed in REVERSE first_bwd_idx order. FX insertion is
    reference-based (we pass the first_bwd node object to inserting_before, not
    an index), so anchor stability isn't strictly necessary — both orderings
    work mechanically. Reverse order is the deterministic convention from the
    course reference and keeps the rewritten graph reading top-to-bottom in the
    same order the picks were chosen.
    """
    if not recomps:
        return gm

    name_to_node: Dict[str, fx.Node] = {n.name: n for n in gm.graph.nodes}

    ordered: List[ActivationMeta] = sorted(
        recomps.values(), key=lambda m: m.first_bwd_idx, reverse=True
    )

    for meta in ordered:
        act = meta.node
        first_bwd = prof.first_backward_access.get(act)

        if first_bwd is None:
            raise RuntimeError(
                f"recomp pick {act.name} has no first_backward_access entry"
            )
        if not meta.recomp_srcs:
            raise RuntimeError(
                f"recomp pick {act.name} has empty recomp_srcs after cascade"
            )

        # Extract subgraph from srcs -> act
        sub = _extract_graph_with_inputs_outputs(
            joint_graph=gm.graph,
            inputs=list(meta.recomp_srcs),
            outputs=[act],
            outputs_descs=[DummyAOTOutput(idx=0)],
            ignore_must_be_in_fw_bw=True,
        )

        local_map: Dict[str, fx.Node] = dict(name_to_node)
        new_act = None

        with gm.graph.inserting_before(first_bwd):
            for n in sub.nodes:
                if n.op == "placeholder" or n.op == "output":
                    continue
                new_node = gm.graph.node_copy(
                    n, arg_transform=lambda arg: local_map[arg.name]
                )
                local_map[n.name] = new_node
                if n.name == act.name:
                    new_act = new_node

        if new_act is None:
            raise RuntimeError(
                f"extraction produced no clone of target activation {act.name}"
            )

        replace_subsequent_uses_of(gm.graph, old_node=act, new_node=new_act)
        name_to_node[new_act.name] = new_act

    gm.graph.lint()
    gm.recompile()
    return gm
