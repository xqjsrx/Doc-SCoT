# run_funsd.py

import os
import json
import re
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

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
#      FUNSD-specific Functions
# ========================================

FUNSD_MAJOR_CATEGORIES = {
    "date", "to", "from", "company", "fax number", "no of stores", "cc", "fax",
    "subject", "note", "manufacturer", "brand", "re", "address", "attn", "fax no",
    "region", "name of account", "type of packings", "test market geography",
    "price point", "sales force involvement", "court", "case name", "lorillard entities",
    "date served", "case type", "comments", "telephone", "message", "client no",
    "message to", "pages including cover sheet", "sender", "reference",
    "number of pages this page included", "sender voice number", "sender fax number",
    "brands applicable", "media type", "coupon value", "circulation",
    "coupon expiration date", "coupon issue date", "code assigned", "est redemption",
    "phone", "other", "c telephone", "product"
}


def load_funsd_test_data(test_data_path):
    """
    加载FUNSD测试数据，格式与LayTextLLM项目一致
    """
    print(f"正在加载FUNSD测试数据: {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # 按图像组织数据
    image_data = {}
    for i, item in enumerate(test_data):
        # 从metadata中提取图像ID
        metadata = json.loads(item.get("metadata", "{}"))
        image_id = metadata.get("image_id", metadata.get("sample_name", "82092117"))  # 默认图像
        # image_filename = f"{image_id}.png"
        image_filename = metadata.get("image")
        
        # 使用image_id作为sample_name
        sample_name = image_id
        
        if sample_name not in image_data:
            image_data[sample_name] = {
                "ocr": item["ocr"],
                "image": image_filename,
                "questions": []
            }
        
        # <<<<< 核心修改点：提取并添加问题类型 >>>>>
        question_text = item["question"]
        entity_name = extract_entity_name(question_text)
        import re
        entity_name = re.sub(r'[^\w\s]', '', entity_name).lower()
        # print("entity_name:",entity_name)

        # 根据预定义的类别列表判断 task_type
        task_type = entity_name if entity_name in FUNSD_MAJOR_CATEGORIES else "others"
        # print("task_type:",task_type)


        # 添加问题
        image_data[sample_name]["questions"].append({
            "task_type": task_type,
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
    # 匹配FUNSD数据集的问题格式
    match = re.search(r'What is the content in the ([^?]+) field?', question)
    if match:
        return match.group(1).strip()
    return question

def normalize_text(text):
    """
    标准化文本用于比较，与LayTextLLM项目中的评估方法一致
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))

def is_correct_prediction(pred, gt):
    """
    判断预测是否正确，与LayTextLLM项目中的评估方法一致
    """
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
        
        # 从问题中提取字段名作为键
        key = result["question"].split("content in the ")[-1].split(" field?")[0].strip('''\"''')
            
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

    # enhanced_question = f"{question} Answer only with the content requested. No other text. Do not include field names or explanations."
    enhanced_question = question

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
    text_prompt = processor.apply_chat_template(messages_with_image, tokenize=False, add_generation_prompt=True)
    
    # 2. 构造 CoT (Phase 2 定义的强序列格式)
    if os.environ.get("EVAL_SELFGEN_COT", "0") == "1":
        # 自生成 CoT 口径：只喂 <think>\n，模型自决定视觉 token 预算并生成 CoT+答案
        full_prompt = text_prompt + "<think>\n"
    else:
        # 固定前缀对照口径：注入 4/4/4 CoT，模型只生成答案
        fixed_cot = build_doc_cot(get_anchor_budget(original_image)) + "\n<answer>"
        full_prompt = text_prompt + fixed_cot
    
    print("模型输入 (" + ("自生成 CoT" if os.environ.get("EVAL_SELFGEN_COT", "0") == "1" else "固定前缀") + " 口径):")
    
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

    is_correct_normal = is_correct_prediction(output_text_normal, gt_text)
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
    model_path = os.environ.get("EVAL_MODEL_PATH", os.path.join(_REPO_ROOT, "output/lora_merged"))

    image_folder_path = os.environ.get("EVAL_IMAGE_FOLDER", os.path.join(_REPO_ROOT, "dataset/FUNSD/images"))

    output_base_path = os.environ.get("EVAL_OUTPUT_BASE", os.path.join(_REPO_ROOT, "eval/result/funsd"))

    funsd_test_data_path = os.environ.get("EVAL_DATA_PATH", os.path.join(_REPO_ROOT, "dataset/FUNSD/funsd_test.json"))
    
    if not os.path.exists(output_base_path): 
        os.makedirs(output_base_path)

    print("正在加载模型...")
    model = CoVTQwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    # 2. 关键：确保模型知道我们要用哪些 Anchor
    model.get_anchor_model_ids(['det', 'layout', 'flow']) 
    
    # 3. 设置 Token ID (用于 Forward 中的 Mask 识别)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 手动获取 ID 并注入模型
    det_id = processor.tokenizer.convert_tokens_to_ids(DET_PAD_TOKEN)
    layout_id = processor.tokenizer.convert_tokens_to_ids(LAYOUT_PAD_TOKEN)
    flow_id = processor.tokenizer.convert_tokens_to_ids(FLOW_PAD_TOKEN)
    
    model.get_anchor_token_idx(
        det_token_idx=det_id,
        layout_token_idx=layout_id,
        flow_token_idx=flow_id
    )
    
    print("模型加载完成，Doc-CoVT 机制已激活。")

    # 加载FUNSD测试数据
    funsd_data = load_funsd_test_data(funsd_test_data_path)
    
    # ==================== [CHECKPOINTING LOGIC - START] ====================

    # <--- 新增: 定义结果文件路径 --->
    normal_summary_path = os.path.join(output_base_path, "funsd_analysis_summary.json")
    
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
    completed_tasks = set()
    for res in normal_results:
        completed_tasks.add((res['sample_name'], res['question']))

    print(f"发现 {len(completed_tasks)} 个已完成的任务。将从断点处继续...")
    
    # <--- 新增: 创建一个待处理所有任务的扁平列表 --->
    all_tasks_to_process = []
    for sample_name, data in funsd_data.items():
        for question_data in data["questions"]:
            all_tasks_to_process.append((sample_name, data, question_data))
            
    # ======================================================================
    
    # <--- 修改: 调整主循环以支持中断继续 --->
    with tqdm(total=len(all_tasks_to_process), desc="Processing FUNSD samples") as pbar:
        pbar.update(len(completed_tasks)) # 设置初始进度

        for sample_name, data, question_data in all_tasks_to_process:

            task_type = question_data["task_type"]
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
            
            try:
                # 获取当前样本的OCR信息
                ocr_info = None
                if USE_OCR_BASE and sample_name in funsd_data:
                    ocr_texts = funsd_data[sample_name]["ocr"]
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
    normal_summary_path = os.path.join(output_base_path, "funsd_analysis_summary.json")
    with open(normal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(normal_results, f, ensure_ascii=False, indent=4)
    print(f"\n推理结果已保存至: {normal_summary_path}")
    
    # 计算准确率
    if normal_results:
        normal_correct_count = sum(1 for r in normal_results if r["is_correct"])
        normal_total_count = len(normal_results)
        normal_accuracy = normal_correct_count / normal_total_count if normal_total_count > 0 else 0
        print(f"\n总体准确率: {normal_correct_count}/{normal_total_count} = {normal_accuracy:.2%}")
        
        # Per-task statistics (normal)
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
            "per_sample_accuracy": normal_accuracy,
            "task_wise_accuracy": normal_task_stats,
            "fscore_metrics": normal_fscore_metrics,
            "total_samples": normal_total_count,
            "correct_samples": normal_correct_count
        }
        
        evaluation_path = os.path.join(output_base_path, "evaluation_results.json")
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        print(f"\n评估结果已保存至: {evaluation_path}")

if __name__ == "__main__":
    main()