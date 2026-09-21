"""从 teacher 信号生成 per-image 锚点预算：每个分支发几个 token，或整步省略。

这是"自适应视觉 token 数量 + 自适应视觉 token 选择"的监督来源。
思路：token 数应该正比于该分支要表达的信息量；信息量平凡时该分支就不该出现。

  k_det    <- 文本量        (word_boxes 数量)
  k_layout <- 版面块数      (DocLayout-YOLO boxes 数量)
  k_flow   <- 阅读顺序复杂度 (按阅读序遍历时 y 坐标回跳次数 ≈ 跨栏/换列次数)

三分支预算取值均为 {0,2,4,6}；k=0 表示该步在 CoT 中整步省略 —— 即分支级选择。
单栏顺排的文档 flow 步会消失，版面平凡的文档 layout 步会消失。

用法:
  python build_anchor_budget.py --cache-dir dataset/teacher_cache/<DS> \
      --out dataset/anchor_budget/<DS>.json
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import torch

MAX_TOTAL = 18  # CoT 里 anchor token 总数上限，控制序列长度


def bucket(v, edges, values):
    """v 落在 edges 的哪一段就取对应 values；edges 递增。"""
    for e, k in zip(edges, values):
        if v <= e:
            return k
    return values[-1]


def flow_wraps(word_boxes, word_ranks):
    """按阅读顺序遍历，统计 y 明显回跳的次数（换列/跨栏）。"""
    if word_boxes is None or word_ranks is None or len(word_ranks) < 3:
        return 0
    order = np.argsort(np.asarray(word_ranks))
    ys = np.asarray(word_boxes)[order][:, 1]
    if len(ys) < 3:
        return 0
    # 阈值取整页高度的 2%，避免同一行内的微小抖动被误计
    thr = 0.02 * max(float(ys.max() - ys.min()), 1e-6)
    return int(np.sum(np.diff(ys) < -thr))


def signals_for(blob):
    """从缓存 blob 提取预算所需的三个原始信号（存这三个数即可，调分段点无需重读 .pt）。"""
    return {
        "n_words": int(len(blob["word_boxes"])) if blob.get("word_boxes") is not None else 0,
        "n_blocks": int(len(blob["boxes"])) if blob.get("boxes") is not None else 0,
        "n_wrap": flow_wraps(blob.get("word_boxes"), blob.get("word_ranks")),
    }


def budget_from_signals(n_words, n_blocks, n_wrap):
    # 分段点依据 25k 图词框分布：<40 真稀疏-收据/标签(30%) / 40~150 中等(50%) / >150 密集(20%)
    k_det = bucket(n_words, [40, 150], [2, 4, 6])
    # 版面只有 0~1 块时结构信息平凡 -> 省略该步
    k_layout = 0 if n_blocks <= 1 else bucket(n_blocks, [3, 8], [2, 4, 6])
    # 无回跳（单栏顺排）-> 阅读顺序平凡 -> 省略该步
    k_flow = 0 if n_wrap == 0 else bucket(n_wrap, [2, 6], [2, 4, 6])

    # 超预算时按比例压缩，det 优先保留（文本定位是最基础的线索）
    total = k_det + k_layout + k_flow
    while total > MAX_TOTAL:
        if k_flow >= k_layout and k_flow > 2:
            k_flow -= 2
        elif k_layout > 2:
            k_layout -= 2
        elif k_det > 2:
            k_det -= 2
        else:
            break
        total = k_det + k_layout + k_flow
    return {"det": k_det, "layout": k_layout, "flow": k_flow,
            "n_words": n_words, "n_blocks": n_blocks, "n_wrap": n_wrap}


def budget_for(blob):
    return budget_from_signals(**signals_for(blob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    print(f"teacher 缓存 {len(files)} 个")
    table, stats = {}, []
    for i, fp in enumerate(files):
        try:
            blob = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception:
            continue
        b = budget_for(blob)
        table[os.path.basename(fp)[:-3]] = {k: b[k] for k in ("det", "layout", "flow")}
        stats.append(b)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(table, open(args.out, "w"))

    n = len(stats)
    print(f"\n完成 {n} 图 -> {args.out}")
    for br in ("det", "layout", "flow"):
        c = Counter(s[br] for s in stats)
        dist = "  ".join(f"k={k}:{v * 100 // n}%" for k, v in sorted(c.items()))
        skip = c.get(0, 0)
        print(f"  {br:<7} 均值 {np.mean([s[br] for s in stats]):.2f}  省略 {skip * 100 // n}%  {dist}")
    tot = [s["det"] + s["layout"] + s["flow"] for s in stats]
    print(f"  总 token 数: 均值 {np.mean(tot):.2f} (固定方案为 12), 范围 {min(tot)}~{max(tot)}")
    combos = Counter(tuple(1 if s[b] else 0 for b in ("det", "layout", "flow")) for s in stats)
    print(f"  分支组合(det,layout,flow 是否出现): {dict(combos)}")


if __name__ == "__main__":
    main()
