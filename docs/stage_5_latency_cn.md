# Stage-5.A.0 延迟 baseline(中文版)

**硬件:** NVIDIA GeForce RTX 3070(8 GB),CUDA 11.8,torch 2.4.1+cu118。
**方法:** 每模式 20 warm-up + 200 计时 forward,batch=1,用 `torch.cuda.Event` 计时。Eager-mode PyTorch(还没上 JIT / TRT)。
**结果文件:** [results/stage5_latency_baseline.csv](../results/stage5_latency_baseline.csv)。
**脚本:** [scripts/stage5_latency_baseline.py](../scripts/stage5_latency_baseline.py)。

英文版:[stage_5_latency.md](stage_5_latency.md)

## 实测每帧延迟(ms)

| 模式 | 参数 | p50 | p90 | p99 | ×M1 |
|---|---:|---:|---:|---:|---:|
| **M1** stage-3.1 单帧                | 12.81 M | 3.15 | 3.41 | 3.49 | 1.00 |
| **M2** stage-3.2 单帧 + DCA          | 13.02 M | 3.57 | 3.79 | 3.95 | 1.13 |
| **M3** stage-3.4 K=10 stateless 时序  | 12.91 M | 3.31 | 3.37 | 3.55 | 1.05 |

**三种模式 RTX 3070 上 p99 都 < 4ms,离 <10ms 论文目标有 ≥6ms 余量。**

## 意外:K-frame 推理只比单帧慢 5%

Stage-5 前的数学推导([REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex §2.2](../REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex))
估算 K=10 stateless 是 **9.0ms** vs 单帧 **4.0ms**(+125%)。**实测 +5%,不是 +125%**。

数学推导假设 K 个 reVAE encode 串行各加 0.50ms。实际 batch=1 时,K 帧 reshape
成 `(K, 1, 96, 160)` minibatch,GPU 一次 kernel dispatch 全部跑完。batch=1
K=10 时 GPU 算力对 0.4 GFLOPS 的 reVAE encoder 严重 under-saturate,**10 个
并行实例只比 1 个贵一点点**。

**含义:** 数学推导里把"stateful vs stateless 决定能否达 <10ms"定为关键的那条
**在 RTX-3070 推理 scale 下其实无所谓**。两种模式都余量充足。等真有 Jetson
实测再重新算。

训练时 stage-3.4 phase 4 看到的 +51% wall 是 batch=16、K=10 → 160 个 effective
minibatch element,**那个**确实把 GPU 跑满。推理 batch=1、K=10 不会。

## Jetson 外推

ResNet-18 在 Jetson Orin NX FP16 上 96×160 输入跑 ~7-10ms/call,**大约比
RTX 3070 慢 2.5×**。线性外推:

| 模式 | RTX 3070 p99 | Jetson Orin NX 预估 p99 |
|---|---:|---:|
| M1 单帧 | 3.49 ms | ~8.7 ms |
| M2 单帧 + DCA | 3.95 ms | ~9.9 ms |
| M3 K=10 stateless | 3.55 ms | ~8.9 ms |

**M1 和 M3 在预估 Jetson FP16 下舒服地在 10ms 预算内。M2 marginal。**

INT8 量化(5.A.3 计划做)通常在 Jetson 上再给 1.5-2× 加速,三种模式都会远在
10ms 之下。

## 关于 <10ms 论文目标的结论

- 延迟目标**对我们训出的三种模型变体(stage-3.1、stage-3.2、stage-3.4)在
  Jetson 级设备上看起来都可达**,只待真 Jetson 实测确认。
- **模型变体的选择应该基于成功率(5.B),不是延迟。**
- 数学推导那个保守的串行 FLOPS 估算**超预测了 K-frame 代价约 25×**。

## 5.A 后续 step

| Sub-step | 状态 | 期望增量 |
|---|---|---|
| 5.A.1 TorchScript trace | 待 | vs eager -10 到 -30% |
| 5.A.2 ONNX 导出 | 待 | TRT 需要的 |
| 5.A.3 TRT FP16 engine | 待 | 再 vs eager -30 到 -50% |
| 5.A.4 最终延迟报告 | 待 | per-platform 表 |

考虑到 eager 模式已经余量舒服,**5.A.1-5.A.3 风险已经被解除** —— 哪怕一般加速
也能把我们推到预算的好几倍以下。

> **更新 2026-05-23**:实机验证已加入 sprint Week 2 计划,5.A.1-3 不再是可选,
> 是 critical path。见 [paper_outline_cn.md](paper_outline_cn.md) §A.3。

## 数学推导需要更新的地方

[`REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex`](../REACT_MATH_Derivations/04_stage5_deployment_math_cn.tex) §2.3
的串行成本表**跟实测不符**。模型在 batch=1 时 under-saturate GPU,所以 K 个
并行 encode 跟 1 个 effective cost 一样。§4 的定性结论("stage-5 先做不会
block 目标;<10ms 不是 binding 约束")依然成立,但理由跟数学推导预测的不同:
**预算比分析模型说的更好达成**。
