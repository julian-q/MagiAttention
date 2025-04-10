import os

import torch
from einops import rearrange

from dffa.benchmarking import Benchmark, do_bench, perf_report
from exps.attn.baselines import attn_impls

# impls = ["sdpa", "fa2", "fa3", "ffa", "torch"]
# impls = ["sdpa", "fa2", "fa3", "ffa"]  # ignore torch native to avoid OOM
impls = ["sdpa", "fa2", "ffa"]  # TODO: support fa3 interface
bs, hs, ss, ds = [1], [8], [1024, 2048, 4096, 6144, 8192, 16384, 32768, 65536], [128]
wds = ["fwd", "bwd"]
quantiles = [0.5, 0.2, 0.8]
bias = None
causal = False
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False
dtype = torch.float16

attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],  # Argument names to use as an x-axis for the plot.
        x_vals=ss,  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=impls,  # Possible values for `line_arg`.
        line_names=impls,  # Label name for the lines.
        styles=[  # Line styles.
            ("green", "--"),
            ("orange", "--"),
            ("steelblue", "--"),
            ("red", "-"),
        ],
        ylabel={  # Label name for the y-axis.
            "flops": "Computation Power (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        plot_name=f"attn-b{b}-h{h}-d{d}-{wd}",  # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "b": b,
            "h": h,
            "d": d,
            "wd": wd,
        },
    )
    for b in bs
    for h in hs
    for d in ds
    for wd in wds
]


@perf_report(attn_flops_configs)
def attn_benchmark(b, h, d, seqlen, wd, provider):
    device = torch.cuda.current_device()

    # flash style shape: (b,s,h,d)
    sq = sk = seqlen
    q = torch.randn(b, sq, h, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=False)

    # sdpa style shape: (b,h,s,d)
    if provider in ("sdpa", "torch"):
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

    # ffa style shape: (t,h,d)
    elif provider == "ffa":
        q = q.view(b * sq, h, d)
        k = k.view(b * sk, h, d)
        v = v.view(b * sk, h, d)

        # ffa args
        assert b == 1, "for now, we only supports b=1 for ffa"
        q_ranges = torch.tensor([[0, sq]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, sk]], dtype=torch.int32, device=device)
        is_causal_mapping = torch.tensor([causal], dtype=torch.bool, device=device)
        max_seqlen_q = sq
        max_seqlen_k = sk

    if wd == "bwd":
        do = torch.randn_like(q)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do]]

    # do the bench
    if provider == "torch":

        def fn():
            return attn_impls.torch_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
                return_attn_probs=return_attn_probs,
            )

        if wd == "bwd":
            o = fn()

            def fn():
                o.backward(do, retain_graph=True)

    elif provider == "fa2":

        def fn():
            return attn_impls.fa2_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                return_attn_probs=return_attn_probs,
            )

        if wd == "bwd":
            o = fn()

            def fn():
                o.backward(do, retain_graph=True)

    elif provider == "fa3":

        def fn():
            return attn_impls.fa3_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        if wd == "bwd":
            o, *rest = fn()

            def fn():
                o.backward(do, retain_graph=True)

    elif provider == "sdpa":

        def fn():
            return attn_impls.sdpa_func(
                q,
                k,
                v,
                is_causal=causal,
                scale=softmax_scale,
                dropout_p=dropout_p,
            )

        if wd == "bwd":
            o = fn()

            def fn():
                o.backward(do, retain_graph=True)

    elif provider == "ffa":

        def fn():
            return attn_impls.ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                is_causal_mapping=is_causal_mapping,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )

        if wd == "bwd":
            o, *rest = fn()

            def fn():
                o.backward(do, retain_graph=True)

    perf_dict = do_bench(
        fn,
        quantiles=quantiles,
        mem_record_mode="peak",
    )

    # post process the perf_dict
    flops_per_matmul = 2.0 * b * h * sq * sk * d
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if wd == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

    def gbps(ms):
        return total_flops / ms * 1e-9

    perf_dict["flops"] = list(map(gbps, perf_dict["flops"]))

    def gb(m):
        return m / 1024**3

    perf_dict["mem"] = list(map(gb, perf_dict["mem"]))

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(script_dir, "outs")
    attn_benchmark.run(print_data=True, save_path=out_root)
