# REACT 数学推导

本目录收录 REACT 在 stage-3.6 之后所有"为什么这样写、为什么不这样写"的数学推导。
每个 `.tex` 文件**自包含**(自带 preamble),可以单独 `pdflatex` 编译,也可以
直接在 GitHub 上阅读 source。

## 文件清单(英文 + 中文双版本)

| 文件 | 内容 | 写作时机 |
|---|---|---|
| [01_collision_loss_saturation.tex](01_collision_loss_saturation.tex) / [01_collision_loss_saturation_cn.tex](01_collision_loss_saturation_cn.tex) | 为什么当前 motion_reshaped_collision_loss 在 single-frame anchor 形态下饱和(stage-3.6 v3 的 ~1% 天花板的数学解释) | stage-3.6 后 |
| [02_multi_waypoint_extension.tex](02_multi_waypoint_extension.tex) / [02_multi_waypoint_extension_cn.tex](02_multi_waypoint_extension_cn.tex) | Option B(多 waypoint GRU decoder)如何把损失从 per-anchor 推广到 per-waypoint;kinodynamic 损失为何在多 waypoint 下"真正激活" | stage-3.6 后 |
| [03_esdf_time_replacement.tex](03_esdf_time_replacement.tex) / [03_esdf_time_replacement_cn.tex](03_esdf_time_replacement_cn.tex) | 把 stage-3.1 `random map_idx` 捷径换成真实动态场景 ESDF 的推导;时空 ESDF 的梯度 | stage-3.6 后 |
| [04_stage5_deployment_math.tex](04_stage5_deployment_math.tex) / [04_stage5_deployment_math_cn.tex](04_stage5_deployment_math_cn.tex) | <10 ms 推理延迟预算分解;K=10 stateless vs stateful GRU 的延迟数学;≥85% 成功率与 dyn_dyn 的(假设)关系 | stage-3.6 后 |
| [05_closedloop_dynamics.tex](05_closedloop_dynamics.tex) / [05_closedloop_dynamics_cn.tex](05_closedloop_dynamics_cn.tex) | 闭环驱动两种 dynamics 的推导:Pass-1 限幅双积分器(SR 上界)vs Pass-2 5 阶多项式(min-jerk + 微分平坦,真机部署同款) | stage-5.B 后 |

## 引用规范

所有公式用 PEMTRS RA-L 2026 论文里的记号(向量小写粗体,矩阵大写,空间索引下标)。

引用上游 YOPO 论文(IEEE RA-L 2024)和 PEMTRS RA-L 2026 的公式编号时,用
`\cite{yopo2024}` / `\cite{pemtrs2026}`。

## 编译

```bash
cd REACT_MATH_Derivations
pdflatex 01_collision_loss_saturation.tex   # 单文件
# 或 mass-compile:
for f in *.tex; do pdflatex "$f"; done
```

LaTeX 依赖:
- 英文版:`amsmath`, `amssymb`, `bm`, `hyperref`, `geometry`(TeX Live full 都自带)
- 中文版:**额外需要 `ctex` 宏包**(`apt install texlive-lang-chinese`
  或 `apt install texlive-full`),并用 `xelatex` 编译以正确处理中文字体:

```bash
xelatex 01_collision_loss_saturation_cn.tex
```
