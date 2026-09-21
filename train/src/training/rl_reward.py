"""
Doc-SCoT: GRPO 奖励函数
- R_answer: 答案正确性（主奖励）
- R_format: 输出格式合规
"""

import re


def normalize_text(text):
    """标准化文本用于比较 (去除空格与标点, 仅适用于精确匹配)"""
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))


def normalize_for_f1(text):
    """
    F1 专用归一化: 保留空格 (否则 split() 后整句变单 token, F1 退化为二值)
    小写 + 标点替换为空格 + 压缩连续空白
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# 语义等价判定用的虚词集 (从语义角度看无信息量的常见词)
_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that',
    'these', 'those', 'there', 'about', 'around', 'approximately', 'roughly',
    'nearly', 'over', 'under', 'between', 'within', 'per', 'by', 'from',
}


def content_words(text):
    """内容词集合: 归一化后去除虚词的 token 集合"""
    tokens = normalize_for_f1(text).split()
    return {t for t in tokens if t not in _STOPWORDS}


def is_semantic_containment(pred_answer, gt_answer):
    """
    判定 pred 是否语义覆盖 GT: GT 的内容词全部出现在 pred 中
    解决"语义已对但啰嗦/截断不一"的重灾区: 答案内容正确但比 GT 多几个修饰词
    """
    gt_words = content_words(gt_answer)
    if not gt_words:
        return False
    pred_words = content_words(pred_answer)
    return gt_words.issubset(pred_words)


def extract_answer(text):
    """从生成文本中提取 <answer>...</answer> 内容"""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: <answer> 之后所有内容
    match = re.search(r'<answer>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: 去掉 </answer> 标签，只保留答案部分
    return text.split('</answer>')[0].strip()


def compute_f1(pred, gt):
    """计算 token 级 F1"""
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def reward_answer(pred_answer, gt_answer):
    """
    答案正确性奖励 (0.0 ~ 1.0)
    三级奖励, 让 RL 优化"语义正确"而非"措辞复刻":
    - 1.0   精确匹配 (激进归一化, 去空格去标点)
    - 0.9   语义包含: GT 内容词 ⊆ pred 内容词
            消解措辞天花板 (诊断结论: 52% 卡住样本是措辞不齐,
            其中 44% 为包含关系), 不再逼模型猜标注者措辞长度
    - 0.8×F1 部分匹配 (连续值, 为组内提供区分度)
    """
    pred_norm = normalize_text(pred_answer)
    gt_norm = normalize_text(gt_answer)

    # 精确匹配
    if pred_norm == gt_norm:
        return 1.0

    # 语义包含: 内容正确且覆盖 GT 信息点, 仅因篇幅超出 GT 而低于精确匹配
    if is_semantic_containment(pred_answer, gt_answer):
        return 0.9

    # 连续 F1 部分奖励 (F1 必须用保留空格的归一化, 否则退化为二值)
    return 0.8 * compute_f1(normalize_for_f1(pred_answer), normalize_for_f1(gt_answer))


def reward_format(generated_text):
    """
    格式合规奖励 (0.0 ~ 1.0)
    检查输出是否包含必要的结构标记
    """
    score = 0.0

    # 有 <answer> 标签
    if '<answer>' in generated_text:
        score += 0.5

    # 有 </answer> 闭合标签
    if '</answer>' in generated_text:
        score += 0.3

    # answer 内容非空
    answer = extract_answer(generated_text)
    if len(answer) > 0:
        score += 0.2

    return score


def compute_reward(generated_text, gt_answer, w_answer=1.0, w_format=0.1):
    """
    综合奖励 = w_answer * R_answer + w_format * R_format

    Args:
        generated_text: 模型生成的完整文本
        gt_answer: 标准答案
        w_answer: 答案奖励权重
        w_format: 格式奖励权重

    Returns:
        reward (float), details (dict)
    """
    pred_answer = extract_answer(generated_text)

    r_ans = reward_answer(pred_answer, gt_answer)
    r_fmt = reward_format(generated_text)

    total = w_answer * r_ans + w_format * r_fmt

    details = {
        'pred_answer': pred_answer,
        'gt_answer': gt_answer,
        'r_answer': r_ans,
        'r_format': r_fmt,
        'reward': total,
    }
    return total, details


MAX_BRANCH_TOKENS = 8   # = query bank 容量；超出的 token 无对应 query，属非法预算


def _cot_spec():
    """(branch, 标签, pad token, 索引槽位)，与 data.build_doc_cot 同源避免格式漂移。

    延迟导入：filter_rl_pool_v3 / screen_hard_samples 只用本模块的文本工具，
    不该被 data.py 的重依赖拖累。
    """
    from .data import DOC_COT_STEPS, _STEP_PAD_TOKEN
    from .constants import ANCHOR_SLOT_TOKENS
    return [(n, t, _STEP_PAD_TOKEN[n], ANCHOR_SLOT_TOKENS[n]) for n, t in DOC_COT_STEPS]


def count_visual_tokens_per_branch(generated_text):
    """<think> 段内各分支实际发出的 token 数——自适应预算下这就是模型的决策量。"""
    think = generated_text.split("</think>")[0]
    return {n: think.count(pad) + sum(think.count(s) for s in slots)
            for n, _, pad, slots in _cot_spec()}


def count_visual_tokens(generated_text):
    """<think> 段内发出的视觉 token 总数。"""
    return sum(count_visual_tokens_per_branch(generated_text).values())


def reward_cot_format(generated_text):
    """自适应 CoT 的语法合规奖励 (0.0 ~ 1.0)。

    变长 CoT 把"发多少 token、省哪一步"变成模型的自由决策，而答案奖励对这些决策几乎
    没有约束力；不加语法项时策略会漂成标签乱序、token 串到别的分支行、或单分支发出远超
    query bank 容量的 token（超出部分没有对应 query，解码侧直接丢弃）。

    五项等权：think 闭合 / 标签不重复且相对顺序合规 / 至少保留一步 /
    视觉 token 全部落在本分支行内 / 每分支 token 数在 [1, MAX_BRANCH_TOKENS] 内。
    """
    spec = _cot_spec()
    if "<think>" not in generated_text:
        return 0.0
    think = generated_text.split("<think>", 1)[1].split("</think>")[0]

    order_ok, present, last_pos = True, [], -1
    for name, label, _, _ in spec:
        n_label = think.count(label)
        if n_label == 0:
            continue
        if n_label > 1:
            order_ok = False
        pos = think.find(label)
        if pos < last_pos:
            order_ok = False
        last_pos = pos
        present.append(name)

    place_ok, counts = True, {}
    for line in think.split("\n"):
        owner = next((n for n, label, _, _ in spec if line.strip().startswith(label)), None)
        for name, _, pad, slots in spec:
            n_tok = line.count(pad) + sum(line.count(s) for s in slots)
            if not n_tok:
                continue
            if name != owner:
                place_ok = False
            else:
                counts[name] = counts.get(name, 0) + n_tok
    tail = generated_text.split("</think>")[-1] if "</think>" in generated_text else ""
    if any(pad in tail or any(s in tail for s in slots) for _, _, pad, slots in spec):
        place_ok = False

    cap_ok = all(1 <= counts.get(n, 0) <= MAX_BRANCH_TOKENS for n in present)

    checks = ["</think>" in generated_text, order_ok, len(present) > 0, place_ok, cap_ok]
    return sum(1.0 for c in checks if c) / len(checks)


def compute_reward_with_cost(generated_text, gt_answer, w_answer=1.0, w_format=0.1,
                             w_token=0.0, token_cap=18, w_cot_fmt=0.0,
                             w_align=0.0, visual_loss=None):
    """答案奖励 + 自适应 CoT 的三个决策项。

    w_token: 视觉 token 开销，越省越好。单独使用时最优解恒为"全省"——诊断已显示视觉
             token 对答案增益微弱，只有开销项时预算必然塌缩到 0。
    w_align: 视觉对齐残差，越准越好。与 w_token 方向相反，两者的均衡点才是这张图真实
             需要的预算。visual_loss 必须在模型侧 charge_absent_branches 打开时计算，
             否则整步省略的残差恒为 0，"全省"重新变成免费最优解。
    w_cot_fmt: 变长 CoT 的语法约束，见 reward_cot_format。
    """
    total, details = compute_reward(generated_text, gt_answer, w_answer, w_format)
    per_branch = count_visual_tokens_per_branch(generated_text)
    n_vis = sum(per_branch.values())
    cost = w_token * (n_vis / max(token_cap, 1))

    r_cot = reward_cot_format(generated_text) if w_cot_fmt else 0.0
    align_cost = w_align * float(visual_loss) if (w_align and visual_loss is not None) else 0.0

    reward = total + w_cot_fmt * r_cot - cost - align_cost
    details.update({
        'n_visual': n_vis,
        'budget': per_branch,
        'token_cost': cost,
        'r_cot_fmt': r_cot,
        'visual_loss': float(visual_loss) if visual_loss is not None else None,
        'align_cost': align_cost,
        'reward': reward,
    })
    return reward, details
