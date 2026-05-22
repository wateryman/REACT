# Stage-5.A.0 Latency Baseline

**Hardware:** NVIDIA GeForce RTX 3070 (8 GB), CUDA 11.8, torch 2.4.1+cu118.
**Method:** 20 warm-up + 200 timed forward passes per mode at `batch=1` via `torch.cuda.Event`.  Eager-mode PyTorch (no JIT / TRT yet).
**Result file:** [results/stage5_latency_baseline.csv](../results/stage5_latency_baseline.csv).
**Script:** [scripts/stage5_latency_baseline.py](../scripts/stage5_latency_baseline.py).

## Measured per-frame latency (ms)

| Mode | params | p50 | p90 | p99 | ×M1 |
|---|---:|---:|---:|---:|---:|
| **M1** stage-3.1 single-frame                | 12.81 M | 3.15 | 3.41 | 3.49 | 1.00 |
| **M2** stage-3.2 single-frame + DCA          | 13.02 M | 3.57 | 3.79 | 3.95 | 1.13 |
| **M3** stage-3.4 K=10 stateless temporal     | 12.91 M | 3.31 | 3.37 | 3.55 | 1.05 |

All three modes finish under 4 ms p99 on RTX 3070 with **>= 6 ms of headroom** to the <10 ms paper target.

## Surprise: K-frame is only 5 % slower than single-frame at inference

The pre-stage-5 math note ([REACT_MATH_Derivations/04_stage5_deployment_math.tex §2.2](../REACT_MATH_Derivations/04_stage5_deployment_math.tex)) projected the K=10 stateless mode at **9.0 ms** vs single-frame at **4.0 ms** (+125 %).  The measurement reports **+5 %**, not +125 %.

The math note assumed each of K reVAE encodes serially adds 0.50 ms.  In reality at `batch=1`, all K frames are reshaped to a `(K, 1, 96, 160)` minibatch that the GPU dispatches as one parallel kernel call.  At batch=1 with K=10, the GPU's compute units are under-saturated for the 0.4 GFLOPS reVAE encoder, so 10 parallel instances cost only marginally more than 1.

**Implication:** the stateful-vs-stateless decision the math note framed as critical for hitting <10 ms is actually **moot at the RTX-3070 inference scale**.  Both modes have ample budget.  We re-visit the calculation when we have an actual Jetson measurement.

The training-time +51 % wall I saw on stage-3.4 phase 4 was at `batch=16, K=10` → 160 effective minibatch elements, which *did* saturate the GPU.  Inference at `batch=1, K=10` does not.

## Jetson extrapolation

Public benchmarks of ResNet-18 forward at 96×160 on Jetson Orin NX FP16 run at ~7-10 ms per call, roughly **2.5× slower than RTX 3070** for the same input.  Linear scaling gives:

| Mode | RTX 3070 p99 | Jetson Orin NX projected p99 |
|---|---:|---:|
| M1 single-frame | 3.49 ms | ~8.7 ms |
| M2 single-frame + DCA | 3.95 ms | ~9.9 ms |
| M3 K=10 stateless | 3.55 ms | ~8.9 ms |

**M1 and M3 are comfortably within the 10 ms budget on projected Jetson FP16.  M2 is marginal.**

INT8 quantization (planned for 5.A.3) typically gives another 1.5-2× speedup on Jetson, putting all three modes well under 10 ms.

## Conclusion for the <10 ms paper target

- The latency target appears **achievable for any of the three model variants** we trained (stage-3.1, stage-3.2, stage-3.4) on a Jetson-class device, modulo the actual Jetson benchmark which we cannot run on the dev box.
- **The choice between model variants should be made on success rate (5.B), not on latency.**
- The math note's conservative serial-FLOPS estimate over-predicted K-frame cost by ~25× the actual measured penalty.

## Next steps within 5.A

| Sub-step | Status | Expected delta |
|---|---|---|
| 5.A.1 TorchScript trace | pending | -10 to -30 % vs eager |
| 5.A.2 ONNX export | pending | needed for TRT |
| 5.A.3 TRT FP16 engine | pending | another -30 to -50 % vs eager |
| 5.A.4 final latency report | pending | per-platform table |

Given the comfortable headroom at eager mode, **5.A.1-5.A.3 are de-risked** — even modest speedups would put us several × under budget.

## What changes in the math derivation

[`REACT_MATH_Derivations/04_stage5_deployment_math.tex`](../REACT_MATH_Derivations/04_stage5_deployment_math.tex) §2.3's serial-cost table **does not match measurement**.  The model under-saturates the GPU at `batch=1`, so K parallel encodes effectively cost as much as 1.  The qualitative conclusion in §4 still stands ("stage-5 first does not block goals; <10 ms is not the binding constraint"), but for a different reason than the note predicted: the budget is *easier to meet* than the analytic model said.
