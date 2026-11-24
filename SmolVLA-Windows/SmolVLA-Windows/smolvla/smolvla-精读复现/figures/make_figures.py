#!/usr/bin/env python3
"""生成 SmolVLA 精读用的教学示意图（中文标注）。

在 lerobot-smolvla-repro 环境运行（有 matplotlib + lerobot），输出到本目录：
  python make_figures.py
图：
  fig-attention-masks.png  三种注意力掩码（训练前向 / 去噪 SA 层 / 去噪 CA 层）
  fig-flow-matching.png    流匹配训练目标与 Euler 采样
  fig-dataflow.png         一次推理的数据流与张量形状
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 与 checkpoint 一致的真实维度 ----
N_IMG_PER_CAM = 64          # 512x512, patch 16 -> 1024 patches, pixel shuffle 4 -> 64
N_CAM = 2
N_LANG = 48                 # tokenizer_max_length
N_STATE = 1
N_ACT = 50                  # chunk_size
D_VLM = 960                 # SmolVLM2-500M text hidden
D_EXP = 720                 # 0.75 * d


def make_att_2d_masks(pad_masks, att_masks):
    """与 lerobot.policies.common.vla_utils.make_att_2d_masks 完全一致的实现。"""
    import torch

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return (att_2d & pad_2d).numpy()[0]


def masks():
    import torch

    n_prefix = N_CAM * N_IMG_PER_CAM + N_LANG + N_STATE
    # 训练前向：图像+语言同一 block（双向），state 另起 block；action token 逐个递增 -> 因果
    pad = torch.ones(1, n_prefix + N_ACT, dtype=torch.bool)
    att = torch.tensor([[0] * (N_CAM * N_IMG_PER_CAM + N_LANG) + [1] + [1] * N_ACT])
    train = make_att_2d_masks(pad, att)
    # 去噪一步：SA 层 keys = prefix(177) + suffix(50)，mask=前缀全可见 + 后缀因果
    sl = N_ACT
    prefix_pad_2d = np.ones((sl, n_prefix), dtype=bool)
    suffix_2d = make_att_2d_masks(
        torch.ones(1, N_ACT, dtype=torch.bool), torch.ones(1, N_ACT, dtype=torch.long)
    )
    sa = np.concatenate([prefix_pad_2d, suffix_2d], axis=1)
    # CA 层 keys 只有 prefix
    ca = prefix_pad_2d
    return train, sa, ca, n_prefix


def draw_mask(ax, m, title, xlabels, ylabels, blocks, vlines=()):
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for b in blocks:
        ax.axvline(b, color="crimson", lw=0.8, ls="--")
        ax.axhline(b, color="crimson", lw=0.8, ls="--")
    for x, t in zip(xlabels[0], xlabels[1]):
        ax.text(x, -0.02 * m.shape[0], t, rotation=90, va="bottom", ha="center", fontsize=7.5)
    for y, t in zip(ylabels[0], ylabels[1]):
        ax.text(-0.02 * m.shape[1], y, t, va="center", ha="right", fontsize=7.5)


def fig_masks():
    train, sa, ca, n_prefix = masks()
    fig = plt.figure(figsize=(13.2, 11.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.0, 1.0], hspace=0.75)
    panels = [
        (train, "(A) 训练前向的完整掩码（227 查询 × 227 键）", 227,
         "前缀内部（相机1/相机2/语言）完全双向；state 行可见前缀但前缀看不到 state（非对称）；动作行只对动作因果。"),
        (sa, "(B) 去噪一步 · 自注意力层 SA（第 0,2,4,…,14 层）：50 查询 × 227 键", 227,
         "键 = 缓存的 177 个前缀 KV + 本步 50 个动作 KV；动作 token i 可见全部前缀 + 自身及之前的动作（因果）。"),
        (ca, "(C) 去噪一步 · 交叉注意力层 CA（第 1,3,5,…,15 层）：50 查询 × 177 键", 177,
         "键只有 177 个前缀 KV（经 k_proj/v_proj 适配）；动作之间完全不可见——这就是“CA 层不做动作交互”。"),
    ]
    for k, (m, title, ncol, note) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1.35, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=10.5)
        if ncol == 227:
            ticks = [32, 96, 152, 176.5, 202]
            labels = ["相机1\n64", "相机2\n64", "语言\n48", "state\n1", "动作\n50"]
        else:
            ticks = [32, 96, 152, 176.5]
            labels = ["相机1\n64", "相机2\n64", "语言\n48", "state\n1"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        for t in (64, 128, 176, 177):
            ax.axvline(t - 0.5, color="crimson", lw=0.7, ls="--", alpha=0.75)
        if k == 0:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)
            for t in (64, 128, 176, 177):
                ax.axhline(t - 0.5, color="crimson", lw=0.7, ls="--", alpha=0.75)
            ax.text(2, 160, "state 列", color="darkgreen", fontsize=8, rotation=90, va="top")
            ax.text(197, 30, "动作列\n（仅对动作\n因果可见）", color="darkred", fontsize=8, ha="center")
        else:
            ax.set_yticks([0, 49])
            ax.set_yticklabels(["动作 0", "动作 49"], fontsize=8)
        ax.text(1.01, 0.5, note, transform=ax.transAxes, fontsize=8.2, va="center", ha="left", wrap=True)
    fig.suptitle("SmolVLA 注意力掩码（数字为 LIBERO checkpoint 实际 token 数：2 相机、48 语言、1 state、50 动作）",
                 fontsize=11.5, y=0.995)
    fig.savefig(os.path.join(HERE, "fig-attention-masks.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_flow():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    # x 轴：t 从 0(干净动作) 到 1(噪声) —— 代码约定 NOISE_AT_ONE
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.6, 0.85)
    ax.axis("off")

    def y_of(step):
        return 0.25 * np.sin(step * 1.1)  # 轻微弯曲，避免重叠

    ts = np.linspace(1.0, 0.0, 11)
    xs = ts
    ax.plot(xs, [y_of(x) for x in xs], color="lightgray", lw=1.2, zorder=0)
    for i, (x, y) in enumerate(zip(xs, [y_of(x) for x in xs])):
        color = "crimson" if i in (0, len(xs) - 1) else "tab:blue"
        ax.scatter([x], [y], s=70 if i in (0, len(xs) - 1) else 38, color=color, zorder=3)
    ax.annotate("x1 = ε（噪声）\nt=1", xy=(1.0, y_of(1.0)), xytext=(0.86, 0.62),
                arrowprops=dict(arrowstyle="->", color="crimson"), fontsize=10, color="crimson")
    ax.annotate("x0 = A（干净动作）\nt=0", xy=(0.0, y_of(0.0)), xytext=(-0.02, 0.56),
                arrowprops=dict(arrowstyle="->", color="crimson"), fontsize=10, color="crimson")
    # 速度场箭头
    for i in range(0, 10, 2):
        x0, x1 = ts[i], ts[i + 1]
        ax.add_patch(FancyArrowPatch((x0, y_of(x0)), (x1, y_of(x1)),
                                     arrowstyle="-|>", mutation_scale=13, color="tab:green", lw=1.6))
    ax.text(0.52, -0.42, "绿箭头 = Euler 步进方向（时间从 t=1 倒着走到 t=0，dt = −1/10）；网络输出 v 本身指向 +t（噪声侧）\n"
                         "x_t = t·ε + (1−t)·A，回归目标 v = ε − A（沿 t 增大方向）；每步 x ← x + dt·v",
            ha="center", fontsize=8.6, color="darkgreen")
    ax.set_title("(a) 训练/采样共用同一条线性概率路径（代码约定：噪声在 t=1）", fontsize=11)

    ax = axes[1]
    t = np.linspace(1e-4, 1, 400)
    from math import gamma

    dens = t ** (1.5 - 1) * (1 - t) ** (1.0 - 1) / (gamma(1.5) * gamma(1.0) / gamma(2.5))
    ax.plot(t, dens, color="tab:purple", lw=2)
    ax.fill_between(t, dens, color="tab:purple", alpha=0.15)
    ax.set_xlabel("代码里的 t（0=干净动作，1=噪声）")
    ax.set_ylabel("Beta(1.5, 1.0) 密度")
    ax.set_title("(b) t ~ Beta(1.5,1.0)×0.999+0.001：训练更常看“接近噪声”的样本", fontsize=11)
    ax.axvline(1.0, color="crimson", ls="--", lw=1)
    ax.text(0.03, ax.get_ylim()[1] * 0.55,
            "论文用 τ 记时间：\nA^τ = τA + (1−τ)ε\n⇒ τ = 1 − t\n论文图/公式里的 τ 与\n代码 t 互为镜像",
            fontsize=9, bbox=dict(fc="lightyellow", ec="gray", alpha=0.9))
    fig.suptitle("SmolVLA 的流匹配：公式（论文 §3.1）↔ 代码（flow_matching.py 的 NOISE_AT_ONE）", fontsize=12)
    fig.savefig(os.path.join(HERE, "fig-flow-matching.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, color, fontsize=8.5, textcolor="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=color, ec="gray", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor, wrap=True)


def arrow(ax, xy0, xy1, color="dimgray"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.2))


def fig_dataflow():
    fig, ax = plt.subplots(figsize=(15.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 输入列
    box(ax, 0.01, 0.80, 0.14, 0.11, "2 路 RGB\n3×512×512", "#dbe9f6")
    box(ax, 0.01, 0.655, 0.14, 0.11, "语言指令\n≤48 token", "#dbe9f6")
    box(ax, 0.01, 0.51, 0.14, 0.11, "state\n8 维 → pad 32", "#dbe9f6")

    # VLM 列
    box(ax, 0.18, 0.665, 0.15, 0.245, "SigLIP 视觉编码\n+ connector\n(pixel shuffle)\n→ 2×64 token ×960", "#cfe3c9", fontsize=8.4)
    box(ax, 0.18, 0.51, 0.15, 0.11, "state_proj\n→ 1×960", "#cfe3c9")
    box(ax, 0.37, 0.51, 0.16, 0.40, "VLM 文本解码器\n前 16 层（共 32）\n\n前缀内双向注意力\n\n输出：KV cache\n+ 前缀隐状态 177×960",
        "#cfe3c9", fontsize=8.4)
    ax.text(0.45, 0.945, "前缀 prefix = 64+64+48+1 = 177 token", ha="center", fontsize=9, color="darkgreen")

    # Euler 迭代框（围绕 action 分支）
    ax.add_patch(FancyBboxPatch((0.155, 0.055), 0.555, 0.275, boxstyle="round,pad=0.008",
                                fc="none", ec="darkorange", lw=1.6, ls="--"))
    ax.text(0.4325, 0.345, "Euler 迭代 10 步：每步 = 一次 Action Expert 前向；每步只算 50 个 action token（前缀 KV 复用）",
            ha="center", fontsize=8.8, color="darkorange")

    # 噪声与 action 分支
    box(ax, 0.01, 0.145, 0.13, 0.11, "噪声 ε\n50×32", "#f6dcdc")
    box(ax, 0.18, 0.145, 0.15, 0.11, "action_in_proj\n+ time_emb MLP\n→ 50×720", "#f6dcdc", fontsize=8.4)
    box(ax, 0.37, 0.125, 0.16, 0.15, "Action Expert\n16 层交错\nSA(0,2,..) / CA(1,3,..)\n宽度 720 = 0.75×960", "#f6e3c1", fontsize=8.2)
    box(ax, 0.575, 0.145, 0.12, 0.11, "action_out_proj\n→ v_θ, 50×32", "#f6dcdc", fontsize=8.4)
    box(ax, 0.74, 0.145, 0.15, 0.11, "x ← x + dt·v\n裁到 7 维 + 反归一化", "#f6dcdc", fontsize=8.4)
    ax.text(0.29, 0.105, "x_t（第 k 步带噪动作）", ha="center", fontsize=8.2, color="darkred")

    # 箭头
    arrow(ax, (0.15, 0.855), (0.18, 0.79))
    arrow(ax, (0.15, 0.71), (0.18, 0.565), color="dimgray")
    arrow(ax, (0.15, 0.565), (0.18, 0.565))
    arrow(ax, (0.33, 0.79), (0.37, 0.79))
    arrow(ax, (0.33, 0.565), (0.37, 0.565))
    arrow(ax, (0.45, 0.51), (0.45, 0.345), color="tab:orange")
    arrow(ax, (0.14, 0.20), (0.18, 0.20))
    arrow(ax, (0.33, 0.20), (0.37, 0.20))
    arrow(ax, (0.53, 0.20), (0.575, 0.20))
    arrow(ax, (0.695, 0.20), (0.74, 0.20))
    # 反馈：输出 v 回到 action_in（体现迭代）
    ax.add_patch(FancyArrowPatch((0.695, 0.17), (0.33, 0.10),
                                 connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                                 mutation_scale=11, color="tab:orange", lw=1.2, ls=":"))
    ax.text(0.52, 0.085, "10 步后得到 x_0 = 干净动作", fontsize=8.2, color="darkorange", ha="center")

    # 右侧要点
    ax.text(0.905, 0.93,
            "关键点\n"
            "① 前缀 prefill 只算一次，\n   10 步去噪复用 KV cache\n"
            "② SA 层：动作之间因果交互；\n   CA 层：只读前缀 KV\n"
            "③ VLM 可冻结（smolvla_base）\n   也可全参微调\n   （官方 LIBERO checkpoint）\n"
            "④ 训练时一次前向算 50 个 token\n   的 MSE，只对 7 个真实维度\n   （padding 到 32 的维度裁掉）",
            fontsize=8.2, va="top", ha="right",
            bbox=dict(fc="white", ec="gray", ls="--", alpha=0.95))
    ax.set_title("SmolVLA 一次推理的数据流（LIBERO checkpoint 实际维度：960 隐藏维、720 expert、chunk 50、10 步 Euler）",
                 fontsize=12, pad=12)
    fig.savefig(os.path.join(HERE, "fig-dataflow.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_masks()
    fig_flow()
    fig_dataflow()
    print("done:", [f for f in os.listdir(HERE) if f.startswith("fig-")])
