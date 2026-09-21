"""数据集 teacher 特征缓存 + token 预算 一键构建。

传入数据集目录（含 data.json），读取 data.json 去重出唯一图片，每张图跑一遍
teacher（DBNet/YOLO/LayoutReader），产出三个文件（均放在数据集目录下）：

  <dataset_dir>/teacher_cache/<图片名>.pt   teacher "原料"缓存（fp16）：
      det_neck [256,128,128]  DBNet FPN 特征（det/flow 解码源）
      det_head [1,512,512]    文本概率图（det GT、词框提取原料）
      layout_feat [384,32,32] YOLO neck 特征（layout 解码源）
      boxes/cls               YOLO 版面框（layout GT、flow 块原料）
      word_boxes/word_ranks   词框与 LayoutReader 词序
      order                   块级阅读顺序（运行时直用）
  <dataset_dir>/budget_signals.json         每图三个原始信号：
      {图片名: {"n_words", "n_blocks", "n_wrap"}} —— 预算的全部依据
  <dataset_dir>/anchor_budget.json          每图 token 预算表：
      {图片名: {"det": k, "layout": k, "flow": k}}，k∈{0,2,4,6}，k=0 整步省略
      由 budget_signals.json 纯查表生成——调整分段点后只需重跑 --budget-only，
      无需重跑 teacher、也无需重读 .pt。

存"原料"而非"成品"：改 loss/GT 光栅化/token 数均不使缓存失效；仅换 teacher 模型或预处理需重算。
断点安全：已存在的 .pt 自动跳过，可重复运行续算；信号与预算表每次运行结束全量重写。

用法:
  PYTHONPATH=train:train/src python tool/build_dataset_cache.py <dataset_dir> [--limit N]
多卡分片（每张卡跑一个 shard，缓存互补）:
  NUM_SHARDS=4 SHARD_ID=i python tool/build_dataset_cache.py <dataset_dir>
  全部 shard 跑完后，任选一卡执行 --budget-only 重建完整预算表：
  python tool/build_dataset_cache.py <dataset_dir> --budget-only
"""
import os
import json
import argparse

import torch
torch.zeros(1).cuda()  # 沙箱环境需先初始化 CUDA
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_anchor_budget import budget_for, signals_for, budget_from_signals

from training.anchor_teachers import AnchorModels, layoutreader_predict_reading_order

CACHE_VERSION = 1  # teacher 组合: DBNet-r50 / DocLayout-YOLO-DocStructBench-1024 / LayoutReader-词级
FLUSH_EVERY = 200


def collect_images(data_json):
    """读 data.json，按出现顺序去重出唯一图片绝对路径。"""
    samples = json.load(open(data_json))
    base_dir = os.path.join(os.path.dirname(data_json), "images")
    uniq, seen = [], set()
    for s in samples:
        p = s["image"] if os.path.isabs(s["image"]) else os.path.join(base_dir, s["image"])
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def run_teacher(am, img):
    """一张图跑一遍三个 teacher，返回缓存 blob（CPU fp16）。"""
    det = am._get_det_outputs(img)
    lay = am._get_layout_outputs(img)
    res = lay["result"]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes_t = res.boxes.xyxyn.cpu()
        cls_t = res.boxes.cls.cpu()
        boxes_l = boxes_t.numpy().tolist()
    else:
        boxes_t = torch.zeros(0, 4)
        cls_t = torch.zeros(0)
        boxes_l = []
    # 块级阅读顺序（词级 LayoutReader 聚合链路，同训练）
    order = am._get_reading_order(img, lay, boxes_l) if boxes_l else []
    # 词级原料（聚合规则变更时可离线重算块序）
    words = am._extract_word_boxes(img)
    word_ranks = []
    if am.layoutreader is not None and len(words) >= 10:
        worder = layoutreader_predict_reading_order(am.layoutreader, words, *img.size)
        word_ranks = [0] * len(words)
        for r, wi in enumerate(worder):
            word_ranks[wi] = r
    return {
        "version": CACHE_VERSION,
        "det_neck": det["neck"][0].half().cpu(),
        "det_head": det["head"][0].half().cpu(),
        "layout_feat": lay["feat"][0].half().cpu(),
        "boxes": boxes_t.float(),
        "cls": cls_t.float(),
        "order": list(order),
        "word_boxes": torch.tensor(words, dtype=torch.float32) if words else torch.zeros(0, 4),
        "word_ranks": torch.tensor(word_ranks, dtype=torch.long) if word_ranks else torch.zeros(0, dtype=torch.long),
    }


def budget_only(dataset_dir, out_json, signals_json):
    """从 budget_signals.json 纯查表生成预算（不碰 .pt，调分段点后重跑此步即可）。
    信号文件缺失时：优先合并各分片信号文件（多卡跑法的收尾），
    都没有时（旧缓存）从 .pt 扫一份信号落盘，再走查表。"""
    if not os.path.exists(signals_json):
        import glob
        shard_files = sorted(glob.glob(os.path.join(dataset_dir, "budget_signals.shard*of*.json")))
        if shard_files:
            print(f"合并 {len(shard_files)} 个分片信号文件...")
            signals = {}
            for fp in shard_files:
                signals.update(json.load(open(fp)))
            json.dump(signals, open(signals_json, "w"))
        else:
            print("无信号文件，从 .pt 扫描生成（一次性迁移）...")
            signals = {}
            for fp in tqdm(sorted(glob.glob(os.path.join(dataset_dir, "teacher_cache", "*.pt")))):
                try:
                    blob = torch.load(fp, map_location="cpu", weights_only=True)
                except Exception:
                    continue
                signals[os.path.basename(fp)[:-3]] = signals_for(blob)
            json.dump(signals, open(signals_json, "w"))
    signals = json.load(open(signals_json))
    table = {name: budget_from_signals(**s) for name, s in signals.items()}
    json.dump(table, open(out_json, "w"))
    print(f"完成 {len(table)} 图 -> {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="数据集目录（含 data.json）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 张（调试用）")
    ap.add_argument("--budget-only", action="store_true",
                    help="不跑 teacher，仅从已有 teacher_cache 重建 anchor_budget.json")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    data_json = os.path.join(dataset_dir, "data.json")
    cache_dir = os.path.join(dataset_dir, "teacher_cache")
    out_json = os.path.join(dataset_dir, "anchor_budget.json")
    signals_json = os.path.join(dataset_dir, "budget_signals.json")
    os.makedirs(cache_dir, exist_ok=True)

    if args.budget_only:
        budget_only(dataset_dir, out_json, signals_json)
        return

    assert os.path.exists(data_json), f"找不到 {data_json}"
    uniq = collect_images(data_json)
    if args.limit:
        uniq = uniq[: args.limit]
    _ns = int(os.environ.get("NUM_SHARDS", "1"))
    _si = int(os.environ.get("SHARD_ID", "0"))
    if _ns > 1:
        uniq = uniq[_si::_ns]
        # 分片各写各的信号文件，避免并发互相覆盖；全部完成后 --budget-only 合并出总表
        signals_json = os.path.join(dataset_dir, f"budget_signals.shard{_si}of{_ns}.json")
        print(f"分片 {_si}/{_ns}")
    print(f"唯一图片: {len(uniq)}")

    am = AnchorModels(["det", "layout", "flow"])
    am.set_device("cuda:0")
    am.set_float()

    # 续算时读入已有信号表，增量更新
    signals = json.load(open(signals_json)) if os.path.exists(signals_json) else {}

    done = skip = fail = 0
    for i, p in enumerate(tqdm(uniq)):
        name = os.path.basename(p)
        out_f = os.path.join(cache_dir, name + ".pt")
        if os.path.exists(out_f):
            skip += 1
            # 缓存已有但信号缺失（旧缓存迁移）：从 .pt 补一份，不触发 teacher 推理
            if name not in signals:
                try:
                    signals[name] = signals_for(torch.load(out_f, map_location="cpu", weights_only=True))
                except Exception as e:
                    print(f"[SIGNAL-FAIL] {p}: {e}")
            continue
        try:
            img = Image.open(p).convert("RGB")
            blob = run_teacher(am, img)
            torch.save(blob, out_f)
            signals[name] = signals_for(blob)
            done += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {p}: {e}")
        if (i + 1) % FLUSH_EVERY == 0:
            json.dump(signals, open(signals_json, "w"))
    json.dump(signals, open(signals_json, "w"))
    print(f"完成 {done}, 跳过(已存在) {skip}, 失败 {fail}")
    print(f"信号 {len(signals)} 图 -> {signals_json}")

    if _ns > 1:
        print(f"分片完成。全部 shard 结束后执行: python tool/build_dataset_cache.py {dataset_dir} --budget-only")
    else:
        table = {name: budget_from_signals(**s) for name, s in signals.items()}
        json.dump(table, open(out_json, "w"))
        print(f"预算表 {len(table)} 图 -> {out_json}")


if __name__ == "__main__":
    main()
