# run_cord_control.py

import os
import json
import re
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- 路径配置 ---
import sys
# 仓库根目录：本文件位于 <repo>/eval/ 下
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "train"))
sys.path.append(os.path.join(_REPO_ROOT, "train", "src"))

from src.training.covt_qwen3_vl import CoVTQwen3VLForConditionalGeneration
from src.training.data import build_doc_cot, get_anchor_budget
from src.training.constants import (
    DET_PAD_TOKEN, 
    LAYOUT_PAD_TOKEN,
    FLOW_PAD_TOKEN,
    ANCHOR_START_TOKEN, ANCHOR_END_TOKEN
)
from transformers import AutoProcessor

# 从共享模块导入通用功能
from shared_utils import (
    setup_seeds,
    resize_image_by_pixel_limit
)

# ========================================
#      CORD-specific Functions
# ========================================

def normalize_text(text):
    """
    标准化文本用于比较，与LayTextLLM项目中的评估方法一致
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))

def load_cord_test_data(test_data_path):
    """
    加载CORD测试数据，格式与LayTextLLM项目一致
    """
    print(f"正在加载CORD测试数据: {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # 按图像组织数据
    image_data = {}
    for item in test_data:
        sample_name = item.get("metadata", {}).get("sample_name") if isinstance(item.get("metadata"), dict) else \
                     json.loads(item.get("metadata", "{}")).get("sample_name", "unknown")
        
        if sample_name not in image_data:
            image_data[sample_name] = {
                "ocr": item["ocr"],
                "image": item.get("image", f"{sample_name}.png"),
                "questions": []
            }
        
        # 不进行任务分类，直接添加问题
        image_data[sample_name]["questions"].append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    print(f"加载了 {len(image_data)} 个图像的数据")
    return image_data

def extract_entity_name(question):
    """
    从问题中提取实体名称
    例如: "What is the \"name of menu\" in the given receipt?" -> "name of menu"
    """
    import re
    # 匹配双引号中的内容
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1)
    # 如果没有双引号，尝试匹配其他模式
    match = re.search(r'What is the ([^?]+) in', question)
    if match:
        return match.group(1).strip()
    return question

# def get_answer_prompt(task_type):
#     """
#     根据任务类型返回相应的回答提示
#     """
#     prompts = {
#         "name of menu": "Please only answer with the name of the first item in the receipt. This refers to the first line item in the detailed bill, not the restaurant or menu name. Look for this information in the upper part of the receipt, not in the summary section at the bottom.",
#         "identification # of menu": "Please only answer with the identification number of the first item. This is the ID number that appears before or alongside the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details.",
#         "unit price of menu": "Please only answer with the unit price of the first item. This is the price per unit of the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details, not in the summary section.",
#         "quantity of menu": "Please only answer with the quantity of the first item. This is the quantity or count of the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details.",
#         "total price of menu": "Please only answer with the total price of the first item. This is the total cost for the first line item (unit price × quantity) in the detailed bill. Look for this information in the upper part of the receipt near the first item details, not in the summary section.",
#         "price of each menu after discount applied": "Please only answer with the price of the first item. This is the final price of the first line item shown in the detailed bill. The answer can be directly obtained from the receipt image without calculation. Look for this information in the upper part of the receipt near the first item details.",
#         "name of submenu": "Please only answer with the name of the first sub-item. Some first items in the receipt consist of multiple sub-items. This refers to the name of the first sub-item that makes up the first line item. The submenu is typically the second line item in the receipt. Look for this information in the upper part of the receipt where sub-item details are listed.",
#         "quantity of submenu": "Please only answer with the quantity of the first sub-item. This is the quantity or count of the first sub-item that makes up the first line item. Look for this information in the upper part of the receipt where sub-item details are listed.",
#         "total price of submenu": "Please only answer with the total price of the first sub-item. This is the total cost of the first sub-item that makes up the first line item. A sub-item is part of a composite item and is typically listed under the main item. The submenu is typically the second line item in the receipt and usually appears directly below the main menu item with a lower price value. Look for this information in the upper part of the receipt where sub-item details are listed. This is NOT the total price of the main menu item.",
#         "subtotal price": "Please only answer with the subtotal price. This is the subtotal amount before tax and service charges are added. Look for this information in the summary section at the bottom of the receipt.",
#         "discounted price in total": "Please only answer with the total discount information. This refers to the discount applied to the entire bill, not the final price after discount. The answer can be directly obtained from the receipt image without calculation. Look for this information in the summary section at the bottom of the receipt. Answer the complete discount field exactly as it appears in the receipt, including all words, numbers, and symbols.",
#         "service charge": "Please only answer with the service charge amount. This is the service fee added to the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "tax amount": "Please only answer with the tax amount. This is the tax applied to the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "others": "Please only answer with the required information.",
#         "total price": "Please only answer with the final total price. This is the grand total amount to be paid. Look for this information in the summary section at the bottom of the receipt.",
#         "amount of price paid in cash": "Please only answer with the amount paid in cash. This is the cash payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of change in cash": "Please only answer with the cash change amount. This is the change returned for cash payment. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of price paid in credit/debit card": "Please only answer with the amount paid by credit or debit card. This is the card payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of price paid in emoney, point": "Please only answer with the amount paid in e-money or points. This is the e-money or point payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "total count of type of menu": "Please only answer with the total count of different item types. This is the number of different types of items in the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "total count of quantity": "Please only answer with the total quantity count. This is the total quantity of all items in the bill. Look for this information in the summary section at the bottom of the receipt. If the answer field contains decimals or units, write the complete value including decimals and units."
#     }
#     return prompts.get(task_type, "Please only answer with the required information.")
    
# def get_answer_prompt(task_type):
#     """
#     根据任务类型返回相应的回答提示
#     """
#     prompts = {
#         "name of menu": "This refers to the first line item in the detailed bill, not the restaurant or menu name. Look for this information in the upper part of the receipt, not in the summary section at the bottom.",
#         "identification # of menu": "This is the ID number that appears before or alongside the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details.",
#         "unit price of menu": "This is the price per unit of the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details, not in the summary section.",
#         "quantity of menu": "This is the quantity or count of the first line item in the detailed bill. Look for this information in the upper part of the receipt near the first item details.",
#         "total price of menu": "This is the total cost for the first line item (unit price × quantity) in the detailed bill. Look for this information in the upper part of the receipt near the first item details, not in the summary section.",
#         "price of each menu after discount applied": "This is the final price of the first line item shown in the detailed bill. The answer can be directly obtained from the receipt image without calculation. Look for this information in the upper part of the receipt near the first item details.",
#         "name of submenu": "Some first items in the receipt consist of multiple sub-items. This refers to the name of the first sub-item that makes up the first line item. The submenu is typically the second line item in the receipt. Look for this information in the upper part of the receipt where sub-item details are listed.",
#         "quantity of submenu": "This is the quantity or count of the first sub-item that makes up the first line item. Look for this information in the upper part of the receipt where sub-item details are listed.",
#         "total price of submenu": "This is the total cost of the first sub-item that makes up the first line item. A sub-item is part of a composite item and is typically listed under the main item. The submenu is typically the second line item in the receipt and usually appears directly below the main menu item with a lower price value. Look for this information in the upper part of the receipt where sub-item details are listed. This is NOT the total price of the main menu item.",
#         "subtotal price": "This is the subtotal amount before tax and service charges are added. Look for this information in the summary section at the bottom of the receipt.",
#         "discounted price in total": "This refers to the discount applied to the entire bill, not the final price after discount. The answer can be directly obtained from the receipt image without calculation. Look for this information in the summary section at the bottom of the receipt. Answer the complete discount field exactly as it appears in the receipt, including all words, numbers, and symbols.",
#         "service charge": "This is the service fee added to the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "tax amount": "This is the tax applied to the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "others": "",
#         "total price": "This is the grand total amount to be paid. Look for this information in the summary section at the bottom of the receipt.",
#         "amount of price paid in cash": "This is the cash payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of change in cash": "This is the change returned for cash payment. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of price paid in credit/debit card": "This is the card payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "amount of price paid in emoney, point": "This is the e-money or point payment amount made for the bill. Look for this information in the payment details section at the bottom of the receipt.",
#         "total count of type of menu": "This is the number of different types of items in the bill. Look for this information in the summary section at the bottom of the receipt.",
#         "total count of quantity": "This is the total quantity of all items in the bill. Look for this information in the summary section at the bottom of the receipt. If the answer field contains decimals or units, write the complete value including decimals and units."
#     }
    return prompts.get(task_type, "")

def get_answer_prompt(task_type):
    return ""


label_mapping = {
    "name of menu": "name of the first menu item",
    "identification # of menu": "identification number of the first menu item",
    "unit price of menu": "unit price of the first menu item",
    "quantity of menu": "quantity of the first menu item",
    "total price of menu": "total price of the first menu item",
    "price of each menu after discount applied": "final price of the first menu item after discount",
    "name of submenu": "name of the first sub-item",
    "quantity of submenu": "quantity of the first sub-item",
    "total price of submenu": "total price of the first sub-item",
    "subtotal price": "subtotal amount",
    "discounted price in total": "total discount amount applied",
    "service charge": "service charge amount",
    "tax amount": "tax amount",
    "total price": "grand total amount",
    "total count of type of menu": "total count of distinct item types",
    "total count of quantity": "total quantity of all items",
    "amount of price paid in cash": "cash payment amount",
    "amount of change in cash": "cash change amount",
    "amount of price paid in credit/debit card": "credit or debit card payment amount",
    "amount of price paid in emoney, point": "e-money or points payment amount",
    "others": "other miscellaneous information"
}

def rewrite_question(question):
    pattern = r'What is the "([^"]+)" in the given receipt\?'
    match = re.search(pattern, question)
    if match:
        original_label = match.group(1)
        if original_label in label_mapping:
            new_label = label_mapping[original_label]
            return question.replace(f'"{original_label}"', f'"{new_label}"')
    return question


def is_correct_prediction(pred, gt, task_type=None):
    """
    判断预测是否正确，与LayTextLLM项目中的评估方法一致
    对于特定任务类型，只比较数字部分
    """
    # 定义需要只比较数字的任务类型
    # numeric_only_tasks = [
    #     "quantity of menu",
    #     "unit price of menu", 
    #     "total count of quantity",
    #     "discounted price in total"
    # ]
    numeric_only_tasks = [
        
    ]
    
    if task_type in numeric_only_tasks:
        # 只提取数字、小数点、逗号和百分号进行比较
        def extract_numeric(text):
            if isinstance(text, list):
                text = text[0] if text else ""
            # 保留数字、小数点、逗号和百分号
            numeric_text = ''.join(c for c in text if c.isdigit() or c in '.%,')
            return numeric_text.lower().strip()
        
        pred_numeric = extract_numeric(pred)
        gt_numeric = extract_numeric(gt)
        return pred_numeric == gt_numeric
    else:
        # 对于其他任务类型，使用原有的标准化方法
        pred_norm = normalize_text(pred)
        gt_norm = normalize_text(gt)
        return pred_norm == gt_norm

def evalFscore(gts, preds):
    """
    实现与LayTextLLM项目一致的F-score计算方法
    """
    # Initialize counts
    total_tp = total_fp = total_fn = 0

    # Iterate through each item
    for key in gts:
        gt_set = {k.strip(): set(v) for k, v in gts[key].items()}
        for item_key, vs in gt_set.items():
            new_vs = []
            for v in vs:
                v = v.lower().translate(str.maketrans("", "", " ().,\n-"))
                new_vs.append(v)
            gt_set[item_key] = set(new_vs)

        pred_set = {k.strip(): set(v) for k, v in preds[key].items()}
        for item_key, vs in pred_set.items():
            new_vs = []
            for v in vs:
                v = v.lower().translate(str.maketrans("", "", " ().,\n-"))
                new_vs.append(v)
            pred_set[item_key] = set(new_vs)

        for label in gt_set:
            # Calculate true positives
            tp_count = sum(1 for gt_item in gt_set[label] if any(gt_item in pred_item for pred_item in pred_set.get(label, [])))
            total_tp += tp_count
            
            # Calculate false positives
            fp_count = sum(1 for pred_item in pred_set.get(label, []) if all(pred_item not in gt_item for gt_item in gt_set[label]))
            total_fp += fp_count

            # Calculate false negatives
            fn_count = sum(1 for gt_item in gt_set[label] if all(gt_item not in pred_item for pred_item in pred_set.get(label, [])))
            total_fn += fn_count

    # Calculate micro precision and recall
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0

    # Calculate micro F1-score
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    print("Micro Precision:", micro_precision)
    print("Micro Recall:", micro_recall)
    print("Micro F1-Score:", micro_f1_score)
    
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1_score": micro_f1_score
    }


def format_for_evaluation(all_results):
    """
    将结果格式化为LayTextLLM评估所需的格式
    """
    gts = {}
    preds = {}
    
    for result in all_results:
        sample_name = result["sample_name"]
        gt = result["ground_truth"]
        pred = result["prediction"]
        
        # CORD数据集使用固定键名
        key = "cord_key"
        # 按照LayTextLLM的格式组织数据
        if sample_name not in gts:
            gts[sample_name] = {key: [gt]}
        else:
            if key not in gts[sample_name]:
                gts[sample_name][key] = [gt]
            else:
                gts[sample_name][key].append(gt)
                
        if sample_name not in preds:
            preds[sample_name] = {key: [pred]}
        else:
            if key not in preds[sample_name]:
                preds[sample_name][key] = [pred]
            else:
                preds[sample_name][key].append(pred)
    
    return gts, preds


def process_single_image_with_gt(model, processor, image_path, question, gt_text, task_type, sample_name, ocr_info=None):
    """
    处理单个图像和任务，包含ground truth 比较
    *** 强制注入 Visual CoT ***
    """
    print(f"\n处理图像: {image_path}")
    print(f"实体: {task_type}")
    print(f"问题: {question}")
    print(f"Ground Truth: {gt_text}")

    # 加载原始图片
    raw_image = Image.open(image_path).convert("RGB")
    original_width, original_height = raw_image.size
    print(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 生成用于模型输入的（可能缩小的）图片
    original_image = resize_image_by_pixel_limit(raw_image)
    if original_image.size != (original_width, original_height):
        print(f"为防止显存溢出，图像已缩放至: {original_image.size}")
    
    print(f"已加载并预处理图像，最终尺寸为: {original_image.size}")

    answer_prompt = get_answer_prompt(task_type)
    enhanced_question = f"{question} {answer_prompt}"

    enhanced_question = rewrite_question(enhanced_question)

    # 构建消息
    content = [
        {"type": "text", "text": enhanced_question},
        {"type": "image", "image": original_image}
    ]
    
    # 如果有OCR信息，则添加到文本提示中
    if ocr_info:
        content[0]["text"] = f"{enhanced_question}\n\n{ocr_info}\n\nPlease base your answer on the provided OCR information above."

    print("模型输入：\n",content[0]["text"])

    messages_with_image = [{"role": "user", "content": content}]
    
    # 1. 生成基础 Prompt (User + System + Assistant Header)
    # 结果类似: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    text_prompt = processor.apply_chat_template(messages_with_image, tokenize=False, add_generation_prompt=True)
    
    
    # 构造 CoT (Phase 2 定义的强序列格式)
    # 注意：这里的换行符 \n 需要和训练数据 data.py 里的逻辑一致
    if os.environ.get("EVAL_SELFGEN_COT", "0") == "1":
        # 自生成 CoT 口径：只喂 <think>\n，模型自决定视觉 token 预算并生成 CoT+答案
        full_prompt = text_prompt + "<think>\n"
    else:
        # 固定前缀对照口径：注入 4/4/4 CoT，模型只生成答案
        fixed_cot = build_doc_cot(get_anchor_budget(original_image)) + "\n<answer>"
        full_prompt = text_prompt + fixed_cot
    
    print("模型输入 (" + ("自生成 CoT" if os.environ.get("EVAL_SELFGEN_COT", "0") == "1" else "固定前缀") + " 口径):")
    # print(full_prompt) # 调试时可以打开
    
    # 4. Tokenize
    inputs_for_gen = processor(text=[full_prompt], images=[original_image], padding=True, return_tensors="pt").to(model.device)
    
    # 5. 推理
    print("正在进行推理 (" + ("自生成 CoT" if os.environ.get("EVAL_SELFGEN_COT", "0") == "1" else "固定前缀") + ")...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs_for_gen, max_new_tokens=int(os.environ.get("EVAL_MAX_NEW_TOKENS", "256")), do_sample=False)
    
    # 6. 解码所有输出
    full_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()
    
    print(f"模型完整输出: '{full_output}'")
    
    # 截取 <answer> 和 </answer> 之间的内容作为答案
    import re
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', full_output, re.DOTALL)
    if match:
        output_text_normal = match.group(1).strip()
    else:
        # 如果没有找到 </answer>，尝试获取 <answer> 之后的所有内容
        match = re.search(r'<answer>\s*(.*)', full_output, re.DOTALL)
        if match:
            output_text_normal = match.group(1).strip()
        else:
            # 如果都没有，使用跳过 input_ids 的方式
            output_text_normal = processor.batch_decode(
                generated_ids[:, inputs_for_gen['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()
    
    print(f"提取的答案: '{output_text_normal}'")

    is_correct_normal = is_correct_prediction(output_text_normal, gt_text, task_type)
    print(f"预测是否正确: {is_correct_normal}")
    
    # 准备返回的结果
    normal_result_data = {
        "image": os.path.splitext(os.path.basename(image_path))[0],
        "task": task_type,
        "question": question,
        "enhanced_question": enhanced_question,
        "prediction": output_text_normal,
        "ground_truth": gt_text,
        "is_correct": is_correct_normal,
        "sample_name": sample_name
    }

    # 输出结果
    print(f"\n=== 结果 ===")
    print(f"Ground Truth: {gt_text}")
    print(f"模型回答: {output_text_normal}")
    print(f"是否正确: {is_correct_normal}")
    print(f"=== 结果保存 ===\n")
    
    return normal_result_data

def main():

    # ==================== 控制开关 ====================
    USE_OCR_BASE = False  # 设置为 True 启用OCR基础功能，False则禁用
    # ==============================================================

    setup_seeds()

    # 全部路径可用环境变量覆盖
    model_path = os.environ.get(
        "EVAL_MODEL_PATH",
        os.path.join(_REPO_ROOT, "output/lora_merged"),
    )

    image_folder_path = os.environ.get(
        "EVAL_IMAGE_FOLDER", os.path.join(_REPO_ROOT, "dataset/CORD/images"))

    output_base_path = os.environ.get(
        "EVAL_OUTPUT_BASE",
        os.path.join(_REPO_ROOT, "eval/result/cord/"),
    )  # 输出根路径

    cord_test_data_path = os.environ.get(
        "EVAL_DATA_PATH", os.path.join(_REPO_ROOT, "dataset/CORD/cord_test.json"))

    if not os.path.exists(output_base_path): 
        os.makedirs(output_base_path)

    print("正在加载模型...")
    model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        # R 方案后 flow 分支为 264 维, 旧 checkpoint 内是 384 维;
        # flow 模块仅训练用, 推理不经过, 忽略尺寸不匹配不影响评测。
        # 注意: 不能与 device_map="auto" 同用, 重新初始化的参数会留在 meta device
        ignore_mismatched_sizes=True,
    ).to("cuda")
    # 2. 关键：确保模型知道我们要用哪些 Anchor
    # 这一步非常重要！因为 Projection Layer 的存在与否可能依赖于这个配置
    # 如果你的 config.json 里没有保存 anchor_model_id，这里必须手动初始化
    # 即使 config 里有，显式调用一次最安全
    model.get_anchor_model_ids(['det', 'layout', 'flow']) 
    
    # 3. 设置 Token ID (用于 Forward 中的 Mask 识别)
    # 必须确保 processor 已经添加了新 token
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 手动获取 ID 并注入模型
    det_id = processor.tokenizer.convert_tokens_to_ids(DET_PAD_TOKEN)
    layout_id = processor.tokenizer.convert_tokens_to_ids(LAYOUT_PAD_TOKEN)
    flow_id = processor.tokenizer.convert_tokens_to_ids(FLOW_PAD_TOKEN)
    
    # 这里的参数列表要看你最新的 covt_qwen3_vl.py 定义
    # 假设你已经更新了 get_anchor_token_idx 方法
    model.get_anchor_token_idx(
        det_token_idx=det_id,       # <--- 注入
        layout_token_idx=layout_id, # <--- 注入
        flow_token_idx=flow_id      # <--- 注入
    )
    
    print("模型加载完成，Doc-SCoT 机制已激活。")

    # 加载CORD测试数据
    cord_data = load_cord_test_data(cord_test_data_path)
    
    # ==================== [CHECKPOINTING LOGIC - START] ====================

    # <--- 新增: 定义结果文件路径 --->
    normal_summary_path = os.path.join(output_base_path, "normal_cord_analysis_summary.json")
    
    # <--- 新增: 检查并加载现有结果 --->
    normal_results = []
    if os.path.exists(normal_summary_path):
        try:
            with open(normal_summary_path, 'r', encoding='utf-8') as f:
                normal_results = json.load(f)
            print(f"已成功加载 {len(normal_results)} 条现有推理结果。")
        except json.JSONDecodeError:
            print(f"警告: {normal_summary_path} 文件损坏，将作为空文件开始。")
            normal_results = []

    # <--- 新增: 创建已完成任务的集合以便快速检查 --->
    # CORD数据集中，一个唯一的任务由 (sample_name, question) 确定
    completed_tasks = set()
    for res in normal_results:
        completed_tasks.add((res['sample_name'], res['question']))

    print(f"发现 {len(completed_tasks)} 个已完成的任务。将从断点处继续...")
    
    # <--- 新增: 创建一个待处理所有任务的扁平列表 --->
    all_tasks_to_process = []
    for sample_name, data in cord_data.items():
        for question_data in data["questions"]:
            all_tasks_to_process.append((sample_name, data, question_data))
            
    # ======================================================================

    # <--- 修改: 调整主循环以支持中断继续 --->
    with tqdm(total=len(all_tasks_to_process), desc="Processing CORD samples") as pbar:
        pbar.update(len(completed_tasks)) # 设置初始进度

        for sample_name, data, question_data in all_tasks_to_process:
            question = question_data["question"]
            
            # 检查任务是否已完成
            if (sample_name, question) in completed_tasks:
                continue

            pbar.set_description(f"Processing {sample_name[:20]}...")

            image_filename = data["image"]
            image_path = os.path.join(image_folder_path, image_filename)
            
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在 {image_path}, skipping.")
                pbar.update(1)
                continue
                
            gt_text = question_data["answer"]
            
            # 直接从问题中提取任务类型
            task_type = extract_entity_name(question)

            try:
                # 获取当前样本的OCR信息
                ocr_info = None
                if USE_OCR_BASE and sample_name in cord_data:
                    ocr_texts = cord_data[sample_name]["ocr"]
                    ocr_info = "OCR information:\n" + "\n".join([f"{text}" for text in ocr_texts])
                
                normal_result = process_single_image_with_gt(
                    model, processor, image_path, question, gt_text, task_type, sample_name, ocr_info=ocr_info
                )

                if normal_result:
                    normal_result["task_type"] = task_type
                    normal_results.append(normal_result)
                    
                    # 增量保存结果
                    with open(normal_summary_path, 'w', encoding='utf-8') as f:
                        json.dump(normal_results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"处理图像 {sample_name} 的问题 {question} 时发生严重错误: {str(e)}")
                
            pbar.update(1)

    print("\n所有任务处理完成。")    
    # 保存所有结果到文件中
    normal_summary_path = os.path.join(output_base_path, "normal_cord_analysis_summary.json")
    with open(normal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(normal_results, f, ensure_ascii=False, indent=4)
    print(f"\n推理结果已保存至: {normal_summary_path}")
    
    # 计算准确率
    if normal_results:
        # 正常推理评估
        normal_correct_count = sum(1 for r in normal_results if r["is_correct"])
        normal_total_count = len(normal_results)
        normal_accuracy = normal_correct_count / normal_total_count if normal_total_count > 0 else 0
        print(f"\n总体准确率: {normal_correct_count}/{normal_total_count} = {normal_accuracy:.2%}")
        
        # 按任务类型分类统计
        normal_task_stats = {}
        for result in normal_results:
            task_type = result["task_type"]
            if task_type not in normal_task_stats:
                normal_task_stats[task_type] = {"correct": 0, "total": 0}
            normal_task_stats[task_type]["total"] += 1
            if result["is_correct"]:
                normal_task_stats[task_type]["correct"] += 1
        
        print("\n按任务类型统计:")
        for task_type, stats in normal_task_stats.items():
            task_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {task_type}: {stats['correct']}/{stats['total']} = {task_accuracy:.2%}")
        
        # 使用LayTextLLM的评估方法计算F-score
        print("\n计算F-score:")
        normal_gts, normal_preds = format_for_evaluation(normal_results)
        normal_fscore_metrics = evalFscore(normal_gts, normal_preds)
        
        # 保存评估结果
        evaluation_results = {
            "normal": {
                "per_sample_accuracy": normal_accuracy,
                "task_wise_accuracy": normal_task_stats,
                "fscore_metrics": normal_fscore_metrics,
                "total_samples": normal_total_count,
                "correct_samples": normal_correct_count
            }
        }
        
        evaluation_path = os.path.join(output_base_path, "evaluation_results.json")
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        print(f"\n评估结果已保存至: {evaluation_path}")

if __name__ == "__main__":
    main()