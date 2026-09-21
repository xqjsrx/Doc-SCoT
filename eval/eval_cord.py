import os
import json
import argparse
from collections import defaultdict

# 仓库根目录：本文件位于 <repo>/eval/ 下
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def load_cord_test_data(test_data_path):
    """
    加载CORD测试数据，获取答案
    """
    print(f"正在加载CORD测试数据: {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # 创建一个以sample_name和question为键的答案字典
    answers = {}
    for item in test_data:
        metadata = json.loads(item.get("metadata", "{}"))
        sample_name = metadata.get("sample_name", "unknown")
        question = item["question"]
        answer = item["answer"]
        answers[(sample_name, question)] = answer
    
    print(f"加载了 {len(answers)} 个答案")
    return answers

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

def format_for_evaluation(results):
    """
    将结果格式化为LayTextLLM评估所需的格式
    """
    gts = {}
    preds = {}
    
    for result in results:
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

def evaluate_output(output_file, test_data_file):
    """
    评估指定格式的output.json文件
    """
    # 加载预测结果
    print(f"正在加载预测结果: {output_file}")
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # 加载测试数据以获取答案
    answers = load_cord_test_data(test_data_file)
    
    # 为每个结果添加答案
    for result in results:
        sample_name = result["sample_name"]
        question = result["question"]
        key = (sample_name, question)
        if key in answers:
            result["ground_truth"] = answers[key]
        else:
            print(f"警告: 找不到样本 {sample_name} 问题 '{question}' 的答案")
            result["ground_truth"] = ""
    
    # 计算准确率
    correct_count = 0
    total_count = len(results)
    
    # 按任务类型统计
    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        task_type = result.get("task_type", "unknown")
        pred = result["prediction"]
        gt = result["ground_truth"]
        
        is_correct = is_correct_prediction(pred, gt)
        result["is_correct"] = is_correct
        
        if is_correct:
            correct_count += 1
            task_stats[task_type]["correct"] += 1
            
        task_stats[task_type]["total"] += 1
    
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n总体准确率: {correct_count}/{total_count} = {overall_accuracy:.2%}")
    
    print("\n按任务类型统计:")
    for task_type, stats in task_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {task_type}: {stats['correct']}/{stats['total']} = {accuracy:.2%}")
    
    # 使用LayTextLLM的评估方法计算F-score
    print("\n计算F-score:")
    gts, preds = format_for_evaluation(results)
    fscore_metrics = evalFscore(gts, preds)
    
    # 保存详细结果
    output_dir = os.path.dirname(output_file)+"/eval_result"
    os.makedirs(output_dir, exist_ok=True)
    detailed_results_file = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n详细结果已保存至: {detailed_results_file}")
    
    # 保存评估摘要
    summary = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_count,
        "correct_samples": correct_count,
        "task_wise_accuracy": dict(task_stats),
        "fscore_metrics": fscore_metrics
    }
    
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"评估摘要已保存至: {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='评估CORD数据集的预测结果')
    parser.add_argument('--output_file', '-o', type=str, 
                        default=os.path.join(_REPO_ROOT, "eval/result/cord/normal_cord_analysis_summary.json"),
                        help='预测结果文件路径')
    
    parser.add_argument('--test_data', '-t', type=str,
                        default=os.path.join(_REPO_ROOT, "dataset/CORD/cord_test.json"),
                        help='测试数据文件路径')
    
    args = parser.parse_args()
    
    evaluate_output(args.output_file, args.test_data)

if __name__ == "__main__":
    main()