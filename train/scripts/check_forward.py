"""End-to-end check of the structural-token path: load the model, inject the anchors, run a
forward pass with the three branch reconstruction losses, backpropagate, and generate.

    MODEL_ID=Qwen/Qwen3-VL-8B-Instruct TEST_IMAGE=/path/to/doc.png \
        python train/scripts/check_forward.py

Exits non-zero if the total loss or the reconstruction loss is not finite, or if no finite
gradient reaches the branch projection heads. Pass --fwd-only to skip backward and generate.
"""
import argparse
import contextlib
import os
import sys
import time

# 仓库根目录：本文件位于 <repo>/train/scripts/ 下
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "train", "src"))

import torch
from transformers import AutoProcessor

from training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from training.constants import DET_PAD_TOKEN, LAYOUT_PAD_TOKEN, FLOW_PAD_TOKEN

BRANCHES = ("det", "layout", "flow")
PAD = {"det": DET_PAD_TOKEN, "layout": LAYOUT_PAD_TOKEN, "flow": FLOW_PAD_TOKEN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--image", default=os.environ.get("TEST_IMAGE"))
    parser.add_argument("--fwd-only", action="store_true",
                        help="forward only; skip backward and generate")
    args = parser.parse_args()
    if not args.image:
        parser.error("--image or TEST_IMAGE is required")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=256 * 32 * 32, max_pixels=512 * 32 * 32)
    tokenizer = processor.tokenizer
    tokenizer.add_tokens(
        ["<think>", "</think>", "<answer>", "</answer>", *(PAD[b] for b in BRANCHES)],
        special_tokens=True)

    model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.resize_token_embeddings(len(tokenizer))

    # 注入各分支的 token id 与 teacher，使 forward 走 anchor loss 分支
    token_ids = [tokenizer(PAD[b], add_special_tokens=False).input_ids[0] for b in BRANCHES]
    model.get_anchor_token_idx(*token_ids)
    model.get_anchor_model_ids(list(BRANCHES))
    print(f"[1/4] model + anchors ready ({time.time() - t0:.1f}s), "
          f"token ids = {dict(zip(BRANCHES, token_ids))}")

    # 固定 4/4/4 的结构块，与 data.build_doc_cot 的输出格式一致
    think = " ".join(f"{b} {PAD[b] * 4}" for b in BRANCHES)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": args.image},
            {"type": "text", "text": "What is this document?"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": f"<think> {think} </think> <answer> a document </answer>"}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    print(f"[2/4] batch built, seq_len={inputs['input_ids'].shape[1]}")

    model.train()
    # 非零步数以取到视觉 loss 权重（衰减调度按 global_steps 计算）
    model.global_steps = 50
    ctx = torch.no_grad() if args.fwd_only else contextlib.nullcontext()
    with ctx:
        out = model(**inputs, labels=labels, image_files=[[args.image]])

    visual_loss = float(out.visual_loss) if out.visual_loss is not None else None
    print(f"[3/4] loss={out.loss.item():.4f}  visual_loss={visual_loss}  "
          f"logits={tuple(out.logits.shape)}")
    assert out.loss is not None and torch.isfinite(out.loss), "total loss is not finite"
    if visual_loss is not None:
        assert visual_loss == visual_loss, "reconstruction loss is NaN"
    if args.fwd_only:
        print("[fwd-only] skipping backward and generate")
        return

    out.loss.backward()
    for branch in BRANCHES:
        grad = getattr(model, f"{branch}_projection").weight.grad
        assert grad is not None and torch.isfinite(grad).all(), \
            f"no finite gradient on {branch}_projection"
        print(f"      {branch}_projection grad |sum| = {grad.abs().sum().item():.6f}")

    model.eval()
    gen_inputs = processor.apply_chat_template(
        [messages[0]], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**gen_inputs, max_new_tokens=32, do_sample=False)
    text = processor.batch_decode(
        gen[:, gen_inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[4/4] generate OK ({time.time() - t0:.1f}s): {text[:100]}")


if __name__ == "__main__":
    main()
