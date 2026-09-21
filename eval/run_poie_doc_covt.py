# run_poie_doc_covt.py

import os
import json
from tqdm import tqdm
import torch
from PIL import Image

# --- 路径配置 ---
import sys
# 仓库根目录：本文件位于 <repo>/eval/ 下
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
# CoT 措辞与结构以 data.py 为唯一来源，避免措辞改动后此处漂移
from src.training.data import build_doc_cot, get_anchor_budget
from transformers import AutoProcessor

# 从共享模块导入通用功能
from shared_utils import (
    setup_seeds,
    resize_image_by_pixel_limit
)

# ========================================
#      POIE-specific Functions
# ========================================

def normalize_text(text):
    """
    标准化文本用于比较，与LayTextLLM项目中的评估方法一致
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))

# 添加标签映射字典
POIE_LABEL_MAPPING = {
    "service size": "SS",
    "calories/energy per serving": "CE-PS",
    "calories/energy per 100g/100ml": "CE-P1",
    "calories/energy percent daily value": "CE-D",
    "calories/energy per 100g percent": "CE-PP",
    "protein per serving": "PRO-PS",
    "protein per 100g/100ml": "PRO-P1",
    "protein percent daily value": "PRO-D",
    "protein per 100g percent": "PRO-PP",
    "total fat per serving": "TF-PS",
    "total fat per 100g/100ml": "TF-P1",
    "total fat percent daily value": "TF-D",
    "total fat per 100g percent": "TF-PP",
    "total carbohydrate per serving": "CAR-PS",
    "total carbohydrate per 100g/100ml": "CAR-P1",
    "total carbohydrate percent daily value": "CAR-D",
    "total carbohydrate per 100g percent": "CAR-PP",
    "sodium per serving": "SO-PS",
    "sodium per 100g/100ml": "SO-P1",
    "sodium percent daily value": "SO-D",
    "sodium per 100g percent": "SO-PP"
}

def load_poie_test_data(test_data_path):
    """
    加载POIE测试数据
    """
    print(f"正在加载POIE测试数据: {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # 按图像组织数据
    image_data = {}
    for item in test_data:
        # 从metadata中提取sample_name
        metadata = json.loads(item.get("metadata", "{}"))
        sample_name = metadata.get("sample_name", "unknown")
        image_filename = metadata.get("image", f"{sample_name}.jpg")
        
        if sample_name not in image_data:
            image_data[sample_name] = {
                "ocr": item["ocr"],
                "poly": item["poly"],
                "image": image_filename,
                "questions": []
            }
        
        # 添加问题和答案
        image_data[sample_name]["questions"].append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    print(f"加载了 {len(image_data)} 个图像的数据")
    return image_data

def get_task_type_from_question(question):
    """
    从问题中提取任务类型并映射到标准标签
    """
    # 遍历映射字典，查找匹配的问题类型
    for key_phrase, label in POIE_LABEL_MAPPING.items():
        if key_phrase in question.lower():
            return label
    return "other"

def get_answer_prompt(task_type):
    """
    根据任务类型返回相应的回答提示
    """
    prompts = {
   }
    return prompts.get(task_type, "Please only answer with the required information.")

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
        # if isinstance(pred, list):
        #     pred = pred[0] if pred else ""
        # if isinstance(gt, list):
        #     gt = gt[0] if gt else ""
        # return pred.lower().translate(str.maketrans("", "", " ().,\n-")) == gt.lower().translate(str.maketrans("", "", " ().,\n-"))
    
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
        
        # POIE数据集按任务类型分类
        task_type = result["task"]
        # 按照LayTextLLM的格式组织数据
        if sample_name not in gts:
            gts[sample_name] = {task_type: [gt]}
        else:
            if task_type not in gts[sample_name]:
                gts[sample_name][task_type] = [gt]
            else:
                gts[sample_name][task_type].append(gt)
                
        if sample_name not in preds:
            preds[sample_name] = {task_type: [pred]}
        else:
            if task_type not in preds[sample_name]:
                preds[sample_name][task_type] = [pred]
            else:
                preds[sample_name][task_type].append(pred)
    
    return gts, preds


def process_single_image_with_gt(model, processor, image_path, question, gt_text, task_type, sample_name, ocr_info=None):
    """
    处理单个图像和任务，包含ground truth 比较
    *** 强制注入 Visual CoT ***
    """
    print(f"\n处理图像: {image_path}")
    print(f"任务: {task_type}")
    print(f"问题: {question}")
    print(f"Ground Truth: {gt_text}")

    # 加载原始图片
    raw_image = Image.open(image_path).convert("RGB")
    # 生成用于模型输入的（可能缩小的）图片
    original_image = resize_image_by_pixel_limit(raw_image)
    print(f"已加载并预处理图像，最终尺寸为: {original_image.size}")

    answer_prompt = get_answer_prompt(task_type)
    enhanced_question = f"{question} {answer_prompt}"

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
        output_text = match.group(1).strip()
    else:
        # 如果没有找到 </answer>，尝试获取 <answer> 之后的所有内容
        match = re.search(r'<answer>\s*(.*)', full_output, re.DOTALL)
        if match:
            output_text = match.group(1).strip()
        else:
            # 如果都没有，使用跳过 input_ids 的方式
            output_text = processor.batch_decode(
                generated_ids[:, inputs_for_gen['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()
    
    print(f"提取的答案: '{output_text}'")

    is_correct = is_correct_prediction(output_text, gt_text, task_type)
    print(f"预测是否正确: {is_correct}")

    # 准备返回结果
    result_data = {
        "image": os.path.splitext(os.path.basename(image_path))[0],
        "task": task_type,
        "question": question,
        "enhanced_question": enhanced_question,
        "prediction": output_text,
        "ground_truth": gt_text,
        "is_correct": is_correct,
        "sample_name": sample_name
    }

    # 输出结果
    print(f"\n=== 结果 ===")
    print(f"Ground Truth: {gt_text}")
    print(f"模型回答: {output_text}")
    print(f"是否正确: {is_correct}")
    print(f"=== 结果保存 ===\n")
    
    return result_data

def main():

    # ==================== 控制开关 ====================
    USE_OCR_BASE = False  # 设置为 True 启用OCR基础功能，False则禁用

    setup_seeds()

    # 全部路径可用环境变量覆盖
    model_path = os.environ.get("EVAL_MODEL_PATH", os.path.join(_REPO_ROOT, "output/lora_merged"))

    image_folder_path = os.environ.get("EVAL_IMAGE_FOLDER", os.path.join(_REPO_ROOT, "dataset/POIE/images"))
    output_base_path = os.environ.get("EVAL_OUTPUT_BASE", os.path.join(_REPO_ROOT, "eval/result/poie"))

    poie_test_data_path = os.environ.get("EVAL_DATA_PATH", os.path.join(_REPO_ROOT, "dataset/POIE/poie_test.json"))
    
    if not os.path.exists(output_base_path): 
        os.makedirs(output_base_path)

    # ==================== [CHECKPOINTING LOGIC - START] ====================

    summary_path = os.path.join(output_base_path, "poie_analysis_summary.json")
    
    results = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"已成功加载 {len(results)} 条现有推理结果。")
        except json.JSONDecodeError:
            print(f"警告: {summary_path} 文件损坏，将作为空文件开始。")
            results = []
    
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
    
    print("模型加载完成，Doc-SCoT 机制已激活。")

    # 加载POIE测试数据
    poie_data = load_poie_test_data(poie_test_data_path)
    
    completed_tasks = set()
    for result in results:
        completed_tasks.add((result["sample_name"], result["question"]))
    
    print(f"发现 {len(completed_tasks)} 个已完成的任务。将从断点处继续...")
    
    all_tasks_to_process = []
    for sample_name, data in poie_data.items():
        for question_data in data["questions"]:
            all_tasks_to_process.append((sample_name, data, question_data))

    # ======================================================================

    with tqdm(total=len(all_tasks_to_process), desc="Processing POIE samples") as pbar:
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
                print(f"警告: 图像文件不存在 {image_path}")
                pbar.update(1)
                continue
                
            gt_text = question_data["answer"]
            task_type = get_task_type_from_question(question)
            
            try:
                # 获取当前样本的OCR信息
                ocr_info = None
                if USE_OCR_BASE and sample_name in poie_data:
                    ocr_texts = poie_data[sample_name]["ocr"]
                    ocr_info = "OCR information:\n" + "\n".join([f"{text}" for text in ocr_texts])
                
                result = process_single_image_with_gt(
                    model, processor, image_path, question, gt_text, task_type, sample_name, ocr_info=ocr_info
                )

                if result:
                    result["task_type"] = task_type
                    results.append(result)
                    
                    # 增量保存结果
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                        
            except Exception as e:
                print(f"处理图像 {sample_name} 的问题 {question} 时发生严重错误: {str(e)}")
            
            pbar.update(1)

    print("\n所有任务处理完成。")
    
    # 保存所有结果到文件中
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n推理结果已保存至: {summary_path}")
    
    # 计算准确率
    if results:
        correct_count = sum(1 for r in results if r["is_correct"])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\n总体准确率: {correct_count}/{total_count} = {accuracy:.2%}")
        
        # 按任务类型分类统计
        task_stats = {}
        for result in results:
            task_type = result["task_type"]
            if task_type not in task_stats:
                task_stats[task_type] = {"correct": 0, "total": 0}
            task_stats[task_type]["total"] += 1
            if result["is_correct"]:
                task_stats[task_type]["correct"] += 1
        
        print("\n按任务类型统计:")
        for task_type, stats in task_stats.items():
            task_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {task_type}: {stats['correct']}/{stats['total']} = {task_accuracy:.2%}")
        
        # 使用LayTextLLM的评估方法计算F-score
        print("\n计算F-score:")
        gts, preds = format_for_evaluation(results)
        fscore_metrics = evalFscore(gts, preds)
        
        # 保存评估结果
        evaluation_results = {
            "per_sample_accuracy": accuracy,
            "task_wise_accuracy": task_stats,
            "fscore_metrics": fscore_metrics,
            "total_samples": total_count,
            "correct_samples": correct_count
        }
        
        evaluation_path = os.path.join(output_base_path, "evaluation_results.json")
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        print(f"\n评估结果已保存至: {evaluation_path}")

if __name__ == "__main__":
    main()