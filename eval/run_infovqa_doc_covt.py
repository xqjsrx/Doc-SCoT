# run_infovqa_doc_covt.py

import os
import json
import re
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np

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
#      InfoVQA-specific Functions
# ========================================

def normalize_text(text):
    """
    标准化文本用于比较，与LayTextLLM项目中的评估方法一致
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))

def load_infovqa_test_data(test_data_path):
    """
    加载InfoVQA测试数据，格式与LayTextLLM项目一致
    """
    print(f"正在加载InfoVQA测试数据: {test_data_path}")
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
                "image": item.get("image", f"{sample_name}.jpeg"),
                "questions": []
            }
        
        # 不进行任务分类，直接添加问题
        image_data[sample_name]["questions"].append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    print(f"加载了 {len(image_data)} 个图像的数据")
    return image_data

def is_correct_prediction(pred, gt):
    """
    判断预测是否正确，与LayTextLLM项目中的评估方法一致
    """
    # 对于InfoVQA，使用原有的标准化方法
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    return pred_norm == gt_norm

# 从LayTextLLM项目导入的评估函数
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normANLS(s1, s2):
    s1 = s1.lower().translate(str.maketrans("", "", " ().,.\n-"))
    s2 = s2.lower().translate(str.maketrans("", "", " ().,.\n-"))
    dist = levenshtein_distance(s1.lower().strip(), s2.lower().strip())
    length = max(len(s1), len(s2))
    value =  0.0 if length == 0 else float(dist) / float(length) 
    return value 

def evaluateANLS(ans_list):
    anls_threshold = 0.5
    anls_list = []
    for predict_pair in ans_list:
        answer = predict_pair["answer"].strip()
        gt_list = predict_pair["annotation"]
        
        value_list = []
        for gt_single in gt_list:
            value_list.append(normANLS(gt_single, answer))
        question_result = 1 - min(value_list)

        if (question_result < anls_threshold) :
            question_result = 0
        anls_list.append(question_result)
    return np.mean(anls_list)

def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
            (ann.lower().translate(str.maketrans("", "", " ().,.\n-")) in elem['answer'].lower().translate(str.maketrans("", "", " ().,.\n-")) ) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    print('sum:', len(scores))
    return sum(scores) / len(scores)

def format_for_anls_accuracy(all_results):
    """
    将结果格式化为ANLS和Accuracy评估所需的格式
    """
    entries = []
    
    for result in all_results:
        gt = result["ground_truth"]
        pred = result["prediction"]
        
        # 按照LayTextLLM的格式组织数据
        entries.append({
            "answer": pred,
            "annotation": gt
        })
    
    return entries

def format_for_cider(all_results):
    """
    将结果格式化为CIDEr评估所需的格式
    """
    res_dict = {}
    gt_dict = {}
    
    for idx, result in enumerate(all_results):
        pred = result["prediction"]
        gt = result["ground_truth"]
        
        # 按照CIDEr的格式组织数据
        res_dict[str(idx)] = [pred]
        gt_dict[str(idx)] = [gt] if isinstance(gt, str) else gt
    
    return res_dict, gt_dict

def process_single_image_with_gt(model, processor, image_path, question, gt_text, sample_name, ocr_info=None):
    """
    处理单个图像和任务，包含ground truth 比较
    *** 强制注入 Visual CoT ***
    """
    print(f"\n处理图像: {image_path}")
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

    # 构建消息
    content = [
        {"type": "text", "text": question},
        {"type": "image", "image": original_image}
    ]
    
    # 如果有OCR信息，则添加到文本提示中
    if ocr_info:
        content[0]["text"] = f"{question}\n\n{ocr_info}\n\nPlease base your answer on the provided OCR information above."

    print("模型输入：\n", content[0]["text"])

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

    is_correct_normal = is_correct_prediction(output_text_normal, gt_text)
    print(f"预测是否正确: {is_correct_normal}")
    
    # 准备返回的结果
    normal_result_data = {
        "image": os.path.splitext(os.path.basename(image_path))[0],
        "question": question,
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

    image_folder_path = os.environ.get("EVAL_IMAGE_FOLDER", os.path.join(_REPO_ROOT, "dataset/InfographicVQA/images"))

    output_base_path = os.environ.get("EVAL_OUTPUT_BASE", os.path.join(_REPO_ROOT, "eval/result/infovqa/"))

    infovqa_test_data_path = os.environ.get("EVAL_DATA_PATH", os.path.join(_REPO_ROOT, "dataset/InfographicVQA/infovqa_test.json"))

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
    
    print("模型加载完成，Doc-CoVT 机制已激活。")

    # 加载InfoVQA测试数据
    infovqa_data = load_infovqa_test_data(infovqa_test_data_path)
    
    # ==================== [CHECKPOINTING LOGIC - START] ====================

    # <--- 新增: 定义结果文件路径 --->
    normal_summary_path = os.path.join(output_base_path, "normal_infovqa_analysis_summary.json")
    
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
    # InfoVQA数据集中，一个唯一的任务由 (sample_name, question) 确定
    completed_tasks = set()
    for res in normal_results:
        completed_tasks.add((res['sample_name'], res['question']))

    print(f"发现 {len(completed_tasks)} 个已完成的任务。将从断点处继续...")
    
    # <--- 新增: 创建一个待处理所有任务的扁平列表 --->
    all_tasks_to_process = []
    for sample_name, data in infovqa_data.items():
        for question_data in data["questions"]:
            all_tasks_to_process.append((sample_name, data, question_data))
            
    # ======================================================================

    # <--- 修改: 调整主循环以支持中断继续 --->
    with tqdm(total=len(all_tasks_to_process), desc="Processing InfoVQA samples") as pbar:
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
            
            try:
                # 获取当前样本的OCR信息
                ocr_info = None
                if USE_OCR_BASE and sample_name in infovqa_data:
                    ocr_texts = infovqa_data[sample_name]["ocr"]
                    ocr_info = "OCR information:\n" + "\n".join([f"{text}" for text in ocr_texts])
                
                normal_result = process_single_image_with_gt(
                    model, processor, image_path, question, gt_text, sample_name, ocr_info=ocr_info
                )

                if normal_result:
                    normal_results.append(normal_result)
                    
                    # 增量保存结果
                    with open(normal_summary_path, 'w', encoding='utf-8') as f:
                        json.dump(normal_results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"处理图像 {sample_name} 的问题 {question} 时发生严重错误: {str(e)}")
                
            pbar.update(1)

    print("\n所有任务处理完成。")    
    # 保存所有结果到文件中
    normal_summary_path = os.path.join(output_base_path, "normal_infovqa_analysis_summary.json")
    with open(normal_summary_path, 'w', encoding='utf-8') as f:
        json.dump(normal_results, f, ensure_ascii=False, indent=4)
    print(f"\n推理结果已保存至: {normal_summary_path}")
    
    # 计算评估指标
    if normal_results:
        # 正常推理评估
        normal_correct_count = sum(1 for r in normal_results if r["is_correct"])
        normal_total_count = len(normal_results)
        normal_accuracy = normal_correct_count / normal_total_count if normal_total_count > 0 else 0
        print(f"\n总体准确率: {normal_correct_count}/{normal_total_count} = {normal_accuracy:.2%}")
        
        # 准备ANLS和Accuracy评估数据
        anls_accuracy_entries = format_for_anls_accuracy(normal_results)
        
        # 计算ANLS
        print("\n计算ANLS:")
        anls_score = evaluateANLS(anls_accuracy_entries)
        print(f"ANLS Score: {anls_score:.2%}")
        
        # 计算Accuracy
        print("\n计算Accuracy:")
        exact_match_accuracy = evaluate_exact_match_accuracy(anls_accuracy_entries)
        print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}")
        
        # 计算CIDEr
        print("\n计算CIDEr:")
        try:
            from pycocoevalcap.cider.cider import Cider
            res_dict, gt_dict = format_for_cider(normal_results)
            cider = Cider()
            cider_score, _ = cider.compute_score(gt_dict, res_dict)
            print(f"CIDEr Score: {cider_score:.4f}")
        except ImportError:
            print("警告: pycocoevalcap 库未安装，无法计算CIDEr分数")
            cider_score = 0.0
        except Exception as e:
            print(f"计算CIDEr时出错: {str(e)}")
            cider_score = 0.0
        
        # 保存评估结果
        evaluation_results = {
            "normal": {
                "per_sample_accuracy": normal_accuracy,
                "exact_match_accuracy": exact_match_accuracy,
                "anls_score": anls_score,
                "cider_score": cider_score,
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