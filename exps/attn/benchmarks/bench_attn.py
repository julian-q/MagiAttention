import os

import torch
from benchmarking import Benchmark, do_bench, perf_report
from benchmarking.attn import attn_impls

benchmark_root = "./benchmarks/"

impls = ["sdpa", "fa2", "sdpa-naive"]
bs, hs, ss, ds = [4], [32], [1024, 2048, 4096, 6144, 8192], [128]
wds = ["fwd", "bwd"]
quantiles = [0.5, 0.2, 0.8]
bias = None
causal = True
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
        styles=[
            ("green", "--"),
            ("orange", "--"),
            ("steelblue", "--"),
            ("red", "-"),
        ],  # Line styles.
        ylabel="tflops",  # Label name for the y-axis.
        plot_name=f"attn-b{b}-h{h}-d{d}-{wd}",  # Name for the plot. Used also as a file name for saving the plot.
        args={
            "b": b,
            "h": h,
            "d": d,
            "wd": wd,
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
    for b in bs
    for h in hs
    for d in ds
    for wd in wds
]


def transpose_func(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


@perf_report(attn_flops_configs)
def attn_benchmark(b, h, d, seqlen, wd, provider):
    device = torch.cuda.current_device()

    # flash style shape: (b,s,h,d)
    sq = sk = seqlen
    q = torch.randn(b, sq, h, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=False)
    if wd == "bwd":
        do = torch.randn_like(q)

    # sdpa style shape: (b,h,s,d)

    qt, kt, vt = map(transpose_func, [q, k, v])
    if wd == "bwd":
        dot = transpose_func(do)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do, qt, kt, vt, dot]]

    # do the bench
    if provider == "sdpa-naive":

        def fn():
            return attn_impls.sdpa_naive_func(
                qt,
                kt,
                vt,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
                return_attn_probs=return_attn_probs,
            )

        # if wd == "bwd":
        #     ot = fn()

        #     def fn():
        #         ot.backward(dot, retain_graph=True)

    elif provider == "fa2":

        def fn():
            attn_impls.fa2_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                return_attn_probs=return_attn_probs,
            )

        # if wd == "bwd":
        #     o = fn()

        #     def fn():
        #         o.backward(do, retain_graph=True)

    elif provider == "sdpa":

        def fn():
            attn_impls.sdpa_func(
                qt, kt, vt, is_causal=causal, scale=softmax_scale, dropout_p=dropout_p
            )

        # if wd == "bwd":
        #     ot = fn()

        #     def fn():
        #         ot.backward(dot, retain_graph=True)

    perf_dict = do_bench(
        fn,
        quantiles=quantiles,
        mem_record_mode="allocated",
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
    attn_benchmark.run(
        print_data=True, save_path=os.path.join(benchmark_root, "./bench_attn_outs/")
    )
