---
layout: distill
permalink: /
title: MagiAttention
description: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Data Training
date: 2025-04-21
featured: true
pretty_table: true
tabs: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

external-links:
  github: https://github.com/SandAI-org/MagiAttention
  arxiv: https://arxiv.org/abs/2505.14135

authors:
  - name: Zewei Tao
    url: "https://github.com/littsk"
    email: zeweitao@sand.ai
    affiliations:
      name: SandAI
  - name: Yunpeng Huang
    url: "https://github.com/Strivin0311"
    email: yunpenghuang@sand.ai
    affiliations:
      name: SandAI, Nanjing University

bibliography: magiattn.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Overview
  - name: Abstract
  - name: Introduction
  - name: Related Work
  - name: Methodology
    subsections:
      - name: Flex-Flash-Attn
      - name: Comp Load-Balance
      - name: Zero-Redundant Comm
      - name: Multi-Stage Overlap
  - name: Experiment
    subsections:
      - name: Kernel-Level
      - name: Module-Level
  - name: Discussion
  - name: Future Work
  - name: FAQ
  - name: Acknowledgement
  - name: Citation

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Overview

<!-- <div class="l-body"> -->
<div class="l-middle">
<!-- <div class="l-page"> -->
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/magiattn_overview_high.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Overview of MagiAttention: (1) FFA, an efficient kernel based on Flash-Attention 3, supports flexible mask patterns; (2) The dispatch solver shards and dispatches packed data with ultra-long contexts and heterogeneous masks, ensuring load-balanced computation; (3) Group-Cast and Group-Reduce primitives eliminate redundant communication; (4) The overlap solver adaptively partitions communication for optimal overlap; (5) During runtime, MagiAttention propagates with flexible and efficient kernels, zero-redundant communication, and multi-stage overlap scheduling, achieving linear scalability.
  </div>
</div>


## Abstract

Training large-scale models for video generation presents two major challenges: (1) The extremely long context length of video tokens, which reaching up to 4 million during training, results in prohibitive computational and memory overhead. (2) The combination of block-causal attention and Packing-and-Padding (PnP) introduces highly complex attention mask patterns. 

To address these challenges, we propose [MagiAttention](https://github.com/SandAI-org/MagiAttention), which aims to support a wide variety of attention mask types with **kernel-level flexibility**, while achieving **linear scalability** with respect to context-parallel (CP) size across a broad range of scenarios, particularly suitable for training tasks involving <u><em>ultra-long, heterogeneous data</em></u> training like video-generation for [Magi-1](https://github.com/SandAI-org/Magi-1).


## Introduction

Training large-scale autoregressive diffusion models like \magi for video generation presents two major challenges: 

- The extremely long context length of video tokens, which reaching up to 4 million during training, results in prohibitive computational and memory overhead. Context-Parallelism (CP) is designed for dealing such long context challenge, but existing state-of-the-art CP methods<d-cite key="jacobs2023deepspeed,liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual"></d-cite> face scalability limitations that face scalability limitations due to size constraints or the high communication overhead inherent in inefficient ring-style point-to-point (P2P) patterns. While recent efforts<d-cite key="wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm"></d-cite> dynamically adjust CP sizes to avoid unnecessary sharding and redundant communication for shorter sequences, they still incur extra memory overhead for NCCL buffers and involve complex scheduling to balance loads and synchronize across different subsets of ranks.

- The combination of block-causal attention and Packing-and-Padding (PnP) introduces highly complex attention mask patterns with variable sequence lengths, which cannot be efficiently handled by existing attention implementations.


To address the aforementioned challenges, we propose MagiAttention, which aims to support a wide variety of attention mask types (\emph{i.e.} kernel flexibility) while achieving linear scalability with respect to context-parallel (CP) size across a broad range of scenarios. Achieving this goal depends on meeting the following fundamental conditions:

- <em>Linearly Scalable Attention Kernel</em>: The performance of the attention kernel should not degradate as CP size increases. To this end, we introduce [Flex-Flash-Attention](#flex-flash-attn), an extension of FlashAttention-3 (FA3), which native considers the efficiency impact of attention mask partitioning in distributed environments. It supports distributable mask representations with a tailored kernel implementation to ensure scalability while accommodating a broader range of attention mask types.
- <em>Balanced Computational Workloads</em>: Imbalances in the computational load across CP ranks lead to unavoidable idle bubbles that hinder scalability. MagiAttention is natively designed to ensure [Computation Load Balancing](#comp-load-balance), mitigating such inefficiencies.
- <em>Full Overlap of Communication and Computation</em>: Without sufficient overlap, increasing CP size results in communication-induced idle time on GPUs, impairing scalability. MagiAttention introduces novel [Zero-Redundant Communication Primitives](#zero-redundant-comm) to minimize communication overhead, along with an [Adaptive Multi-Stage Overlap](#multi-stage-overlap) strategy that enables effective communication-computation overlap.

The overview of MagiAttention is shown in [Overview](#overview), and we will introduce key designs in the following [Methodology](#methodology) section, with comprehensive experimental results presented in [Experiment](#experiment).


## Related Work

To tackle the ultra-long context challenge in large-scale model training, the distributed attention mechanism, or context parallelism (CP), is essential. 

However, current strategies fall short in our demanding settings. DeepSpeed’s Ulysses<d-cite key="jacobs2023deepspeed"></d-cite> leverages the multi-head characteristic for head-sharded, sequence-complete propagation in the attention module, and head-complete, sequence-sharded propagation elsewhere, with transformation between parallel placements efficiently handled by All-to-All collective communication. While easy to integrate, it has scalability limitations, requiring the number of heads to be divisible by the CP size, particularly in GQA settings or with tensor parallelism<d-cite key="shoeybi2020megatronlm,korthikanti2022reducing"></d-cite>. In contrast, Ring-Attention<d-cite key="li2021sequence,liu2023ringattentionblockwisetransformers,wang2024tokenringefficientparallelismframework"></d-cite> keeps sequence-sharded activations and accesses the complete sequence through multi-stage ring-style point-to-point (P2P) communication to perform online attention propagation<d-cite key="rabe2021self,dao2022flashattention"></d-cite> and pipeline compute-communication overlap<d-cite key="wang2022overlap"></d-cite>. Though more scalable, it suffers from significant communication overhead due to large communication volumes and inefficient P2P send/receive primitives over the entire CP group as the CP size increases. Some following works<d-cite key="fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual"></d-cite> combine Ulysses and Ring-Attention in a 2D distributed attention approach to mitigate their limitations, yet still lack the efficiency and scalability required for our ultra-long context settings.

Worse still, for irregular attention mask patterns like the aforementioned varlen masks, classic Ring-Attention-based CP strategies are facing more challenges, besides the attention kernel limitations. First, the naive <em>sequential even sharding</em> along the sequence dimension causes uneven distribution of the varlen mask area, leading to imbalanced computational loads across CP ranks. Although the customized <em>zigzag sharding</em> design<d-cite key="ring_flash_attention_issue2"></d-cite> balances loads for specific (varlen) causal mask patterns in the following figure, it results in kernel performance degradation from fragmented sharding and excessive padding, and does not generalize well to other patterns including the <em>varlen block-causal mask</em> met in autoregressive video generation.


<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ring_attn_load_balance.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Illustration of Ring-Attention’s customized sharding strategies for load balancing. (a) Full mask uses sequential sharding for the global mask; (b) Causal mask employs tailored <em>zigzag sharding</em><d-cite key="ring_flash_attention_issue2"></d-cite>; (c) Varlen full mask applies sequential sharding per local mask (one per packed sample); (d) Varlen causal mask uses <em>zigzag sharding</em> per local mask, causing performance degradation from fragmentation and padding.
  </div>
</div>

Second, the communication overhead issue is exacerbated under sparse varlen mask settings, as entire sequence chunks are still transferred across all CP ranks even when not all ranks require them, might causing over 30% redundant communication costs as illustrated in [Zero-Redundant Comm](#zero-redundant-comm). Third, the former challenges cause the pipeline compute-communication overlap strategy fails more often due to imbalanced loads and large communication overheads, further limiting scalability.

Recent efforts<d-cite key="wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm"></d-cite> attempt to address these issues by dynamically assigning communication groups of specific CP sizes to different samples based on their sequence lengths, to reduce unnecessary sharding and redundant communication for shorter sequences. However, these methods suffer from extra memory overhead for NCCL buffers and complex scheduling for computation load-balance and synchronization across different sets of ranks.


## Methodology

### Flex-Flash-Attn


Flash Attention<d-cite key="dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention"></d-cite> is foundational in large-scale model training for its superior performance and support for varlen-packed data. However, it offers limited support for irregular attention masks, particularly when such patterns are distributed across CP ranks, resulting in increased complexity and underscoring the need for a more flexible attention kernel<d-cite key="pytorch_sdpa, dong2024flexattentionprogrammingmodel,wang2025flashmaskefficientrichmask"></d-cite> without compromising performance.

Therefore, we introduce Flex-Flash-Attention (FFA), which is natively designed for distribution scenarios and provides greater flexibility in handling diverse attention mask types. The core idea behind FFA is to generalize a <b>distributable</b> formulation for irregular attention masks by decomposing the entire mask into multiple computational units, each referred to as an $\mathrm{AttnSlice}$. Each $\mathrm{AttnSlice}$ is defined by a triplet $\mathrm{(QRange, KRange, MaskType)}$, which specifies a submask with a basic shape bounded by a contiguous 2D query-key region as seen in the figure below.


<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/attnslice_interpret.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Illustration of $\mathrm{AttnSlice}$ formulation for some irregular mask. It decomposes the original mask into multiple $\mathrm{AttnSlice}$s and allows re-expression of fractal masks after rearrangement across CP ranks, making it suitable for distributed attention. Note that computation load balance across CP ranks is not considered in this illustration.
  </div>
</div>


Using this formulation, as shown in the figure below, a wide variety of commonly used attention masks, including the varlen block-causal mask for autoregressive video generation, can be expressed as a composition of multiple such triplets even after sharding and rearrrangement in distributed settings, making FFA highly suitable for distributed attention computation.


<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/mask_with_attn_slice.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Examples of mask patterns formulated by $\mathrm{AttnSlice}$. (a)-(d) Standard FA3-compatible patterns; (e)-(h) Irregular masks beyond Flash-Attention's capabilities, including the varlen block-causal mask, which FFA supports seamlessly while maintaining performance comparable to FA3.
  </div>
</div>


Built on Flash-Attention 3 (FA3) kernels<d-cite key="shah2024flashattention3fastaccurateattention"></d-cite>, Flex-Flash-Attention (FFA) leverages Hopper GPUs' TMA feature<d-cite key="nvidia2024accelerating"></d-cite> and introduces slice-level parallelism with atomic operations for correctness as illustrated in the following figure, achieving comparable MFU to FA3 while supporting the flexible $\mathrm{AttnSlice}$ formulation (see [Kernel-Level Experiments](#kernel-level) for FFA performance and flexibility benchmarks compared to other attention kernels).

### Comp Load-Balance

In context-parallel settings, different CP ranks may be assigned heterogeneous attention masks, resulting in imbalanced computational workloads across ranks. Ring-Attention, as mentioned in [Related Work](#related-work), employs a specialized partitioning strategy designed specifically for causal attention, which limits its applicability to more general attention patterns. To overcome this limitation, we propose a generic and efficient dispatch solver that enables balanced workload distribution across CP ranks for a broad range of attention types.

First, to enable finer-grained control, we propose a chunk-wise permutable sharding strategy as seen in [Overview](#overview). Specifically, the entire mask is evenly partitioned along the query-dimension into chunks, each associated with a submask area: $\lbrace(C_i, \mathrm{Area}(C_i))\rbrace_{i=1}^n$, where $C_i$ indicates i-th chunk, $\mathrm{Area}(C_i)$ is the mask area of $C_i$, $n$ is $\frac{seqlen}{\textit{chunk_size}}$, and $\textit{chunk_size}$ is a hyperparameter controlling granularity. 

These chunks are then equally assigned to $\textit{cp_size}$ buckets, with each bucket containing the exact same number of chunks to ensure token-level load balance in non-attention modules, attaching with a summed submask area, denoted as $\lbrace(B_j, \mathrm{SumArea}(B_j))\rbrace_{j=1}^{\textit{cp_size}}$.


With above strategy, we could fine-grained control the computational workloads of each CP rank, and the load-balancing dispatch becomes a combinatorial optimization problem, defined as finding an optimal mapping function $f^*: \lbrace C_i\rbrace_{i=1}^n \rightarrow \lbrace B_j\rbrace_{j=1}^{\textit{cp_size}}$ follows:

$$
\begin{aligned}
    &f^* = \arg \min\limits_{f}\max\limits_{j}\left\{\mathrm{SumArea}(B_j)\right\} \label{eq:comp_load_balance}\\
    &\text{s.t.}\;\;|B_j| = \frac{n}{\textit{cp_size}}, \;\; seqlen \;\%\; (\textit{cp_size} \times \textit{chunk_size}) = 0\nonumber
\end{aligned}
$$

However, this optimization is a known NP-hard problem, making it impractical to find an optimal solution on-the-fly during each training iteration, especially given the varying mask patterns across micro-batches. Thus, we propose an efficient greedy algorithm as shown below that provides a suboptimal yet effective solution within $O(n\log n)$ complexity.

<div class="l-body">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/min_hp_alg.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Greedy Load-Balance Dispatch Algorithm via Min-Heap
  </div>
</div>

### Zero-Redundant Comm

The existing ring-style implementation uses point-to-point send/recv communication primitives, which cannot provide sufficient communication granularity, resulting in redundant communication. Take causal mask as an example, we analyze the redundant communication by recording the distribution of remote key-value ($\mathrm{KV}$) requests and their gradients ($\mathrm{dKV}$) under sparse attention masks. As shown in the following figure, $\mathrm{KV}_0$ is required by all queries and should be sent to all devices via Broad-Cast in the forward pass, with $\mathrm{dKV}_0$ reduced via All-Reduce in the backward pass. In contrast, $\mathrm{KV}_7$ is only needed by its host device but still circulates through all devices, and this redundancy intensifies in varlen scenarios.


<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ring-p2p-redundancy.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Examples illustrating redundant communication in Ring P2P patterns for distributed attention given heterogeneous masks.: (a) Even with a simple causal mask, Ring P2P incurs <b>25%</b> redundant communication; (b) For irregular mask patterns such as varlen block-causal mask with last global block, Ring P2P results in over <b>33%</b> redundancy.
  </div>
</div>

To address this, as illustrated in the figure below, we introduce two communication primitives: $\textit{Group-Cast}$ and $\textit{Group-Reduce}$, which model the communication patterns of low-demand $\mathrm{KV}$ and $\mathrm{dKV}$. For example, in the causal mask, $\mathrm{KV}_5$ on $\mathrm{rank}_2$ is required only by $\{\mathrm{Q}_6,\mathrm{Q}_7\}$ and should be sent exclusively to the target ranks $\{\mathrm{rank}_0, \mathrm{rank}_1\}$ via Group-Cast, while the partial $\mathrm{dKV}_5$ is collected and reduced back to $\mathrm{rank}_2$ via Group-Reduce accordingly.

<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/group-gather-reduce-all2allv.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Illustration of Group-Cast/Group-Reduce primitives for zero redundancy, using the varlen block-causal mask with the last global block as an example for irregular patterns. (a) In both forward and backward passes, the Group-Cast primitive internally analyzes and generates a transfer table for $\mathrm{KV}$ send/receive buffers, and launches the underlying All-to-All-v to complete communication with our custom $\texttt{Range Gather}$ kernel for pre-/post-processing. (b) In the backward pass, Group-Reduce similarly handles the partial $\mathrm{dKV}$ communication for reduction, using All-to-All-v with the \texttt{Range Gather} kernel for pre-processing and the $\texttt{Range Scatter-Reduce}$ kernel for post-processing.
  </div>
</div>

As no existing communication kernels support these primitives, we prototype them using All-to-All-v, achieving zero-redundant communication in both forward and backward passes. However, this approach introduces extra pre-/post-processing overhead, similar to (un)permutation in expert parallelism (EP)<d-cite key="gale2022megablocks"></d-cite>. While kernel fusion mitigates the overhead, a dedicated implementation of Group-Cast and Group-Reduce remains a key direction for future work.


### Multi-Stage Overlap

Leveraging previous optimizations, we achieve high-performance computation through an efficient kernel and balanced workload dispatch, while minimizing communication overhead with our new primitives. To drive true linear scalability, we further improve end-to-end performance by introducing a multi-stage compute-communication overlap strategy, that effectively hides communication latency and adaptively optimizes overlap through manual or automatic tuning.

Similar to prior works<d-cite key="liu2023ringattentionblockwisetransformers,zhao2023pytorch,async_tensor_parallelism_in_pytorch"></d-cite>, we schedule pipeline stages to overlap computation with communication for both forward and backward passes, as shown in the following figureFig. Each $\mathrm{rank}_i$ first partitions its remote $\mathrm{KV}$/$\mathrm{dKV}$ communication into stages. 

<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/multi_stage_overlap_fwd_bwd.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Schematic of Magi Attention's multi-stage overlap scheduling. (a) Forward pass: 4-stage scheduling overlaps computation (partial attention outputs and $\textit{lse}$ factors) with prefetching of next-stage $\mathrm{KV}$ requests (where applicable), hiding all communication overhead with the final stage's computation exposed. (b) Backward pass: 3-stage scheduling overlaps computation (partial $\mathrm{dQ}$, $\mathrm{dKV}$) with prefetching of next-stage $\mathrm{KV}$ requests and reduction of prior $\mathrm{dKV}$ requests, hiding all communication overhead except the $\mathrm{dKV}$ reduction of the final stage.
  </div>
</div>

In the forward pass, the scheduler first launches the Group-Cast kernel to prefetch the next remote $\mathrm{KV}$, then asynchronously executes the FFA kernel for partial attention computation, hiding all communication behind computation. To prevent all SMs from being occupied by the attention kernel, by default, we ensure the communication kernel picked first by setting `CUDA_DEVICE_MAX_CONNECTIONS=1`<d-cite key="cuda_device_max_connections_issue"></d-cite>. However, we also support relax this constraint by setting an non-zero `sm_margin` argument for the FFA kernel, to preserve some SMs for communication kernels to be launched.


In the backward pass, besides prefetching the next $\mathrm{KV}$, the Group-Reduce kernel reduces the last $\mathrm{dKV}$ in a separate CUDA stream before launching the FFA kernel for the current stage, ensuring communication is overlapped across all stages except the final $\mathrm{dKV}$ reduction. Due to PyTorch's one-to-one mapping for process groups and collective communication streams including All-to-All-v<d-cite key="collectives_nccl_stream_issue"></d-cite>, we internally use an additional CP group for Group-Reduce to enable full overlap between communication kernels in the backward pass.

To adaptively control overlap granularity, we further introduce a tunable hyperparameter, $\texttt{num_stages}$, accounting for varying compute-to-communication ratios across training setups, microbatches, or between forward and backward passes. This parameter can be manually configured or automatically determined by our $\textit{overlap solver}$, with a simple dynamic search algorithm as shown below.

<div class="l-body">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/dynamic_mso_alg.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Dynamic Overlap Stage Search Algorithm
  </div>
</div>


## Experiment

### Kernel-Level

To demonstrate FFA kernels' state-of-the-art performance and flexibility in handling ultra-long, heterogeneous data training, we measure the computing power (in $\texttt{TFLOPs/s}$) on Hopper GPUs for both forward and backward passes of prevalent attention kernels across standard and irregular mask patterns.

| settings              | value                                                                          |
|-----------------------|-----------------------------------------------------------------------------|
| batch size (b)        | 1                                                                            |
| number of heads (nh)  | nhq:nhk:nhv = 48:8:8 (GQA)                                    |
| head dimension (hd)   | 128                                                                           |
| dtype                 | torch.bfloat16                                                               |
| dropout probability   | 0.0                                                                          |
| window size           | 1024 (for sliding window masks only)                        |

Benchmark settings: for each mask pattern, we vary the sequence length $seqlen$ from $4k,8k,16k,...,$ up to $128k$ ($seqlen_q = seqlen_k = seqlen$) while measuring computation power (in $\texttt{TFLOPs/s}$) for forward and backward passes of different attention kernels. Other configurations are fixed using common training settings (see the table above) to focus on the impact of sequence length and mask pattern.

Results are reported in the following figures.

<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ffa_perf_report_full_all_family.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Benchmarking FFA's performance and flexibility against other leading attention kernels for full mask scenarios.
  </div>
</div>

<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ffa_perf_report_causal_all_family.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Benchmarking FFA's performance and flexibility against other leading attention kernels for causal mask scenarios.
  </div>
</div>



### Module-Level


To validate the linear scalability of MagiAttention, we assess the computing power (in $\texttt{TFLOPs/s}$) of the attention module propagation as the sequence length and parallel size increases for both forward and backward passes across various mask patterns, and compare it with several state-of-the-art CP strategies.

The experiments are conducted on a large-scale productive GPU cluster<d-footnote>Due to business and confidentiality reasons, specific details about the productive cluster, such as the number and type of GPUs, are withheld.</d-footnote>. We scale the total sequence length $\textit{seqlen}$, the context-parallel size $\textit{cp_size}$, and the node size $\textit{nnodes}$ together from $(\textit{seqlen}:64k, \textit{cp_size}:1, nnodes:1)$, $(\textit{seqlen}:128k, \textit{cp_size}:2, nnodes:2)$, ..., to $(\textit{seqlen}:3840k (\sim 4M), \textit{cp_size}:60, nnodes:60)$. 

The tensor-parallel size $\textit{tp_size}$ is fixed at 8, with sequence-parallel enabled. Other data and model configurations for different mask types are the same as in the table in [Kernel-Level Experiments](#kernel-level).

Therefore, in every training setting, each rank is assigned a constant 64k $\textit{seqlen}$ for attention propagation, while the $\textit{seqlen}$ of remaining activations stays 8k. This setup simulates a common training configuration.

The results are presented in the following figures.


<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ffa_perf_report_full_all_family.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Benchmarking FFA's performance and flexibility against other leading attention kernels for full mask scenarios.
  </div>
</div>

<div class="l-middle">
  <div class="row mt-3">
      <div class="col-sm mt-3 mt-md-0">
          {% include figure.liquid loading="eager" path="assets/img/magiattn/ffa_perf_report_causal_all_family.png" class="img-fluid rounded z-depth-1" zoomable=true %}
      </div>
  </div>
  <div class="caption">
    Benchmarking FFA's performance and flexibility against other leading attention kernels for causal mask scenarios.
  </div>
</div>


## Discussion



## Future Work



## FAQ


## Acknowledgement

We are grateful to the contributors listed below for their valuable contributions during the early stages of MagiAttention.

| Member   | Affiliations         | Email                        | GitHub Account    |
|:-----------|:-------------|:----------------------------|:---------------|
| Zewei Tao    | SandAI       | zeweitao@sand.ai            | littsk         |
| Yunpeng Huang    | SandAI, Nanjing University       | yunpenghuang@sand.ai,hyp@smail.nju.edu.cn       | Strivin0311    |
| Qiangang Wang    | Nanjing University | 522024330081@smail.nju.edu.cn | WT1W           |
| Hanwen Sun   | SandAI, Beijing University |  shw20010329@163.com |  hanwen-sun  |
| Tao Bu      | Nanjing University | 502024330002@smail.nju.edu.cn | Big-TRex       |
| WenYang Fang    | Nanjing University | fwy@smail.nju.edu.cn        | kagami4243     |
| Siyuang Yan    | Nanjing University | siyuanyan@smail.nju.edu.cn  | FibonaccciYan  |
| Zixu Jiang     | Nanjing University | 522023330040@smail.nju.edu.cn | 191220042      |
| Dingkun Xu    | Nanjing University | 211220090@smail.nju.edu.cn  | PureDimension  |
| Mingyu Liang    | Nanjing University |   mingyuliang518@gmail.com     | gaomusiki      |
| Jingwei Xu    | Nanjing University | jingweix@nju.edu.cn | paragonlight   |


## Citation

If you use MagiAttention in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Data Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```

