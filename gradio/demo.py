import time
import torch
from PIL import Image
import gradio as gr
import os
import sys
import numpy as np
import cv2
from sklearn.decomposition import PCA

# --- 路径配置 ---
# 仓库根目录：本文件位于 <repo>/gradio/ 下
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "train"))
sys.path.append(os.path.join(_REPO_ROOT, "train", "src"))

from src.training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from src.training.constants import (
    DET_PAD_TOKEN, 
    LAYOUT_PAD_TOKEN,
    FLOW_PAD_TOKEN,
    ANCHOR_START_TOKEN, ANCHOR_END_TOKEN
)
from transformers import AutoProcessor

# ================= Configuration Area =================
# 合并完 LoRA 的最终权重路径；可用 MODEL_PATH 环境变量覆盖
DEFAULT_MODEL_NAME = os.environ.get(
    "MODEL_PATH", os.path.join(_REPO_ROOT, "output/lora_merged"))
# ======================================================

_cached_model = None
_cached_processor = None

def load_model_and_processor(model_name: str, ckpt: str = None):
    path_to_load = ckpt if ckpt is not None else model_name
    print(f"Loading model from: {path_to_load}")

    model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        path_to_load,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        path_to_load,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True
    )

    print("Loading ALL Anchor Models... WARNING: High VRAM usage!")
    try:
        # === 修改点 2: 在列表中加入 'det' 以加载 DBNet ===
        model.get_anchor_model_ids(['det', 'layout', 'flow'])
        
        model.anchor_models.set_device(model.device)
        # model.anchor_models.set_float() 
    except Exception as e:
        print(f"Warning: Failed to load Anchor Models. Error: {e}")

    return model, processor

def get_cached_model_and_processor(model_name=DEFAULT_MODEL_NAME):
    global _cached_model, _cached_processor
    if _cached_model is not None and _cached_processor is not None:
        return _cached_model, _cached_processor
    _cached_model, _cached_processor = load_model_and_processor(model_name)
    return _cached_model, _cached_processor

def decode_visual_features(model, processor, image, input_ids, hidden_states):
    """
    通用解码器
    # 返回: (seg_img, depth_img, edge_img, dino_img, det_img, layout_img)
    # 返回: (edge_img, det_img, layout_img, flow_img)
    返回: (det_img, layout_img, flow_img)
    """
    # === 修改点 3: 增加 res_det, res_layout 和 res_flow 初始化 ===
    # res_seg, res_depth, res_edge, res_det, res_layout = None, None, None, None, None
    # res_edge, res_det, res_layout, res_flow = None, None, None, None
    res_det, res_layout, res_flow = None, None, None
    device = model.device

    # === 修改点 4: Detection (Text Heatmap) ===
    try:
        det_token_id = processor.tokenizer.convert_tokens_to_ids(DET_PAD_TOKEN)
        det_mask = (input_ids == det_token_id)
        
        # [诊断 1] 看看有没有检测到 det token
        if not det_mask.any():
            print("[DEBUG] No <|det_pad|> tokens found in output!")
            
        if det_mask.any():
            # 1. 提取隐层
            feat = hidden_states[0, det_mask] 
            
            # 2. 投影 & Cross-Attention
            proj = torch.nn.functional.normalize(model.det_projection(feat).unsqueeze(0))
            query = model.det_query_vectors.unsqueeze(0).to(proj.dtype)
            det_attn_out, _ = model.det_cross_attention(query, proj, proj)
            
            # [诊断 2] 打印 VLM 生成的 Token 统计信息
            print(f"[DEBUG] VLM Tokens: Mean={det_attn_out.mean().item():.4f}, Std={det_attn_out.std().item():.4f}")
            
            # 3. 提取 Teacher 特征
            det_embed = model.anchor_models.get_det_embed(image)
            
            if det_embed is not None:
                det_embed = det_embed.to(det_attn_out.dtype)
                
                # [诊断 3] 打印 Teacher 特征统计信息 (关键!)
                # 如果这里全是 0，说明 Doctr Hook 失败了
                print(f"[DEBUG] Teacher Feats: Mean={det_embed.mean().item():.4f}, Std={det_embed.std().item():.4f}, Max={det_embed.max().item():.4f}")
                
                # 4. 解码
                # [作弊模式] 绕过 Token，直接看 Teacher 特征图是否包含文字信息
                # 如果这个图能显示文字，说明 Teacher 是好的，是 Token 没学好
                # pred_map = torch.mean(det_embed, dim=1, keepdim=True) # 直接平均特征通道
                
                # [正常模式]
                pred_map = model.anchor_models.decode_det_tokens(det_attn_out, det_embed)
                
                # 5. 转图 (Heatmap)
                heatmap = pred_map.squeeze().float().detach().cpu().numpy() # [128, 128]
                
                # === [关键修复] 动态对比度拉伸 (Min-Max Normalization) ===
                # 能够把微弱的 0.49~0.51 的差异拉伸到 0~1，让结构显形
                h_min, h_max = heatmap.min(), heatmap.max()
                print(f"[DEBUG] Detection Map Range: Min={h_min:.4f}, Max={h_max:.4f}, Mean={heatmap.mean():.4f}")
                
                if h_max - h_min > 1e-6: # 防止除以零
                    heatmap = (heatmap - h_min) / (h_max - h_min)
                else:
                    # 如果真的是纯色，保持原样（会显示为单色）
                    pass
                
                # 归一化到 0-255
                heatmap = (heatmap * 255).astype(np.uint8)
                
                # 使用 JET 颜色映射
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # 转回 RGB 并 Resize
                res_det = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
                res_det = res_det.resize(image.size, resample=Image.BILINEAR)

    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"Detection Decode Error: {e}")

    # === 修改点 6: Layout (Document Layout Analysis) ===
    try:
        layout_token_id = processor.tokenizer.convert_tokens_to_ids(LAYOUT_PAD_TOKEN)
        layout_mask = (input_ids == layout_token_id)
        
        if not layout_mask.any():
            print("[DEBUG] No <|layout_pad|> tokens found in output!")
            
        if layout_mask.any():
            feat = hidden_states[0, layout_mask] 
            
            proj = torch.nn.functional.normalize(model.layout_projection(feat).unsqueeze(0))
            query = model.layout_query_vectors.unsqueeze(0).to(proj.dtype)
            layout_attn_out, _ = model.layout_cross_attention(query, proj, proj)
            
            print(f"[DEBUG] Layout VLM Tokens: Mean={layout_attn_out.mean().item():.4f}, Std={layout_attn_out.std().item():.4f}")
            
            layout_embed = model.anchor_models.get_layout_embed(image)
            
            if layout_embed is not None:
                layout_embed = layout_embed.to(layout_attn_out.dtype)
                
                print(f"[DEBUG] Layout Teacher Feats: Mean={layout_embed.mean().item():.4f}, Std={layout_embed.std().item():.4f}, Max={layout_embed.max().item():.4f}")
                
                pred_map = model.anchor_models.decode_det_tokens(layout_attn_out, layout_embed)
                
                heatmap = pred_map.squeeze().float().detach().cpu().numpy() 
                
                h_min, h_max = heatmap.min(), heatmap.max()
                print(f"[DEBUG] Layout Map Range: Min={h_min:.4f}, Max={h_max:.4f}, Mean={heatmap.mean():.4f}")
                
                if h_max - h_min > 1e-6:
                    heatmap = (heatmap - h_min) / (h_max - h_min)
                else:
                    pass
                
                heatmap = (heatmap * 255).astype(np.uint8)
                
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                res_layout = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
                res_layout = res_layout.resize(image.size, resample=Image.BILINEAR)

    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"Layout Decode Error: {e}")

    # === 修改点 8: Flow (Reading Flow) ===
    try:
        flow_token_id = processor.tokenizer.convert_tokens_to_ids(FLOW_PAD_TOKEN)
        flow_mask = (input_ids == flow_token_id)
        
        if not flow_mask.any():
            print("[DEBUG] No <|flow_pad|> tokens found in output!")
            
        if flow_mask.any():
            feat = hidden_states[0, flow_mask] 
            
            proj = torch.nn.functional.normalize(model.flow_projection(feat).unsqueeze(0))
            query = model.flow_query_vectors.unsqueeze(0).to(proj.dtype)
            flow_attn_out, _ = model.flow_cross_attention(query, proj, proj)
            
            print(f"[DEBUG] Flow VLM Tokens: Mean={flow_attn_out.mean().item():.4f}, Std={flow_attn_out.std().item():.4f}")
            
            flow_embed = model.anchor_models.get_flow_embed(image)
            
            if flow_embed is not None:
                flow_embed = flow_embed.to(flow_attn_out.dtype)
                
                print(f"[DEBUG] Flow Teacher Feats: Mean={flow_embed.mean().item():.4f}, Std={flow_embed.std().item():.4f}, Max={flow_embed.max().item():.4f}")
                
                pred_map = model.anchor_models.decode_det_tokens(flow_attn_out, flow_embed)
                
                heatmap = pred_map.squeeze().float().detach().cpu().numpy() 
                
                h_min, h_max = heatmap.min(), heatmap.max()
                print(f"[DEBUG] Flow Map Range: Min={h_min:.4f}, Max={h_max:.4f}, Mean={heatmap.mean():.4f}")
                
                if h_max - h_min > 1e-6:
                    heatmap = (heatmap - h_min) / (h_max - h_min)
                else:
                    pass
                
                heatmap = (heatmap * 255).astype(np.uint8)
                
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                res_flow = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
                res_flow = res_flow.resize(image.size, resample=Image.BILINEAR)

    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"Flow Decode Error: {e}")

    # === 修改点 9: 返回包含 res_flow ===
    # return res_seg, res_depth, res_edge, res_det, res_layout
    # return res_edge, res_det, res_layout, res_flow
    return res_det, res_layout, res_flow

def run_single_inference(model, processor, image, question, max_new_tokens=512, **kwargs):
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
        image_ref = image
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        image_ref = "gradio_image"
    else:
        raise ValueError("Invalid image input")

    # Part 1: Text Generation
    messages = [{"role": "user", "content": [{"type": "image", "image": image_ref}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # prompt += "<think>" 
    inputs = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Running Text Generation...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, **kwargs
        )
    end = time.time()
    
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=True)

    # === Decode Spontaneous Thoughts ===
    print("Decoding Autonomous Tokens...")
    with torch.no_grad():
        full_pass_outputs = model(
            input_ids=outputs,
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs['image_grid_thw'],
            output_hidden_states=True,
            return_dict=True
        )
        spon_hidden_states = full_pass_outputs.hidden_states[-1]
    
    spon_imgs = decode_visual_features(
        model, processor, pil_image, outputs[0], spon_hidden_states
    ) 

    # Part 2: Forced Visualization
    print("Running Forced Visualization...")
    # === 修改点 10: 在 Prompt 中加入 DET_PAD_TOKEN, LAYOUT_PAD_TOKEN 和 FLOW_PAD_TOKEN ===
    anchor_seq = (
        ANCHOR_START_TOKEN + 
        DET_PAD_TOKEN * 4 +
        LAYOUT_PAD_TOKEN * 4 +
        FLOW_PAD_TOKEN * 4 +  # <--- 新增
        ANCHOR_END_TOKEN
    )
    forced_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Visualize everything.<|im_end|>\n<|im_start|>assistant\n{anchor_seq}"
    
    forced_inputs = processor(text=[forced_prompt], images=[pil_image], return_tensors="pt")
    forced_inputs = {k: v.to(model.device) for k, v in forced_inputs.items()}

    with torch.no_grad():
        forced_outputs = model(**forced_inputs, output_hidden_states=True, return_dict=True)
        forced_hidden_states = forced_outputs.hidden_states[-1]
        forced_ids = forced_inputs['input_ids'][0]

    forced_imgs = decode_visual_features(
        model, processor, pil_image, forced_ids, forced_hidden_states
    )

    print("\n" + "="*50)
    print("[模型输出结果]")
    print("="*50)
    print(f"文本答案: {answer}")
    print(f"推理时间: {end - start:.2f} 秒")
    print(f"\n自主视觉思考:")
    # print(f"  - 分割图: {'已生成' if spon_imgs[0] is not None else 'None'}")
    # print(f"  - 深度图: {'已生成' if spon_imgs[1] is not None else 'None'}")
    # print(f"  - 边缘图: {'已生成' if spon_imgs[0] is not None else 'None'}")
    print(f"  - 检测图: {'已生成' if spon_imgs[0] is not None else 'None'}")
    print(f"  - 布局图: {'已生成' if spon_imgs[1] is not None else 'None'}")
    print(f"  - 阅读流图: {'已生成' if spon_imgs[2] is not None else 'None'}")
    print(f"\n强制可视化:")
    # print(f"  - 分割图: {'已生成' if forced_imgs[0] is not None else 'None'}")
    # print(f"  - 深度图: {'已生成' if forced_imgs[1] is not None else 'None'}")
    # print(f"  - 边缘图: {'已生成' if forced_imgs[0] is not None else 'None'}")
    print(f"  - 检测图: {'已生成' if forced_imgs[0] is not None else 'None'}")
    print(f"  - 布局图: {'已生成' if forced_imgs[1] is not None else 'None'}")
    print(f"  - 阅读流图: {'已生成' if forced_imgs[2] is not None else 'None'}")
    print("="*50 + "\n")

    return answer, end - start, *spon_imgs, *forced_imgs

def gradio_inference(image, question, max_new_tokens, temperature, top_p, seed):
    if image is None:
        # === 修改点 11: 返回值数量(8) ===
        # return "Please upload an image.", 0.0, None, None, None, None, None, None, None, None, None
        return "Please upload an image.", 0.0, None, None, None, None, None, None, None

    model, processor = get_cached_model_and_processor()

    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # === 修改点 12: 解包 3+3=6 张图 ===
    # spon_imgs 和 forced_imgs 现在各包含 3 张图 (Det, Layout, Flow)
    result = run_single_inference(
        model=model,
        processor=processor,
        image=image,
        question=question,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=(temperature > 0.0),
    )
    
    # result 结构: answer, elapsed, spon(3), forced(3)
    return result

def build_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CoDP Demo: Document Visual Thinking")
        # gr.Markdown("Visualizing Segmentation, Depth, Edge, Semantics, and **Text Detection**.")
        gr.Markdown("Visualizing Text Detection, Document Layout, and Reading Flow.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input", type="pil")
                question_input = gr.Textbox(label="Question", value="Describe the scene in detail.")

                with gr.Accordion("Advanced Options", open=False):
                    max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=1024, value=512)
                    temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.0, value=0.0)
                    top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9)
                    seed = gr.Slider(label="seed", minimum=0, maximum=1000, value=42)
                
                gr.Markdown("### Example")
                example_image_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "assets", "LayoutLMv3.png")
                )
                example_image = Image.open(example_image_path).convert("RGB")
                gr.Examples(
                    examples=[
                        [
                            example_image,
                            # "Describe the scene in the picture in detail, and find out how many clouds are in the sky. Use segmentation, depth map, edge map, and perception feature information of the image to answer this question."
                            "What is the title of this paper. Use text detection map, document layout map, and reading flow map of the image to answer this question."
                        ]
                    ],
                    inputs=[image_input, question_input],
                    examples_per_page=1
                )

                run_button = gr.Button("Run Inference", variant="primary")
            
            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="Text Answer", lines=5)
                elapsed_output = gr.Number(label="Time (s)")

        # === 修改点 13: 更新 UI 列数 (4 -> 5) ===
        gr.Markdown("### 🤖 Autonomous Visual Thoughts")
        with gr.Row():
            # spon_seg = gr.Image(label="Auto: Seg", type="pil")
            # spon_depth = gr.Image(label="Auto: Depth", type="pil")
            # spon_edge = gr.Image(label="Auto: Edge", type="pil")
            spon_det = gr.Image(label="Auto: Detection (Doc)", type="pil")
            spon_layout = gr.Image(label="Auto: Layout (Doc)", type="pil")
            spon_flow = gr.Image(label="Auto: Reading Flow (Doc)", type="pil") # 新增

        gr.Markdown("### 🛠️ Forced Visual Capabilities")
        with gr.Row():
            # forced_seg = gr.Image(label="Forced: Seg", type="pil")
            # forced_depth = gr.Image(label="Forced: Depth", type="pil")
            # forced_edge = gr.Image(label="Forced: Edge", type="pil")
            forced_det = gr.Image(label="Forced: Detection (Doc)", type="pil")
            forced_layout = gr.Image(label="Forced: Layout (Doc)", type="pil")
            forced_flow = gr.Image(label="Forced: Reading Flow (Doc)", type="pil") # 新增

        run_button.click(
            fn=gradio_inference,
            inputs=[image_input, question_input, max_new_tokens, temperature, top_p, seed],
            outputs=[
                answer_output, elapsed_output, 
                # spon_seg, spon_depth, 
                # spon_edge, spon_det, spon_layout, spon_flow,      # 4个自主
                spon_det, spon_layout, spon_flow,      # 3个自主
                # forced_edge, forced_det, forced_layout, forced_flow # 4个强制
                forced_det, forced_layout, forced_flow # 3个强制
            ]
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)