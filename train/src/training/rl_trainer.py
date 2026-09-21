"""
Doc-SCoT: GRPO (Group Relative Policy Optimization) 训练器

核心流程:
1. 对每个 (image, question) 采样 G 条回答
2. 用奖励函数评分
3. 组内标准化得到优势
4. GRPO loss = -mean(advantage * log_prob) + beta * KL
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image

from .rl_reward import compute_reward, compute_reward_with_cost, count_visual_tokens
from .data import build_doc_cot, get_anchor_budget
from .constants import (
    DET_PAD_TOKEN, LAYOUT_PAD_TOKEN, FLOW_PAD_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    VISION_START_TOKEN, VISION_END_TOKEN, DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
)


class RLDataset(Dataset):
    """RL 训练数据集: (image_path, question, gt_answer)"""

    def __init__(self, data_path, image_folder):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item['image']
        if isinstance(image_file, list):
            image_file = image_file[0]
        image_path = os.path.join(self.image_folder, image_file)

        conversations = item['conversations']
        question = conversations[0]['value'].replace('<image>\n', '').replace('<image>', '')
        gt_answer = conversations[1]['value']

        return {
            'image_path': image_path,
            'question': question,
            'gt_answer': gt_answer,
        }


def build_rl_prompt(question, processor, force_cot_prefix=True, image_file=None):
    """
    构建 RL 推理 prompt
    - force_cot_prefix=True: 注入 CoT 前缀（与SFT训练格式一致），模型生成答案文本
    - force_cot_prefix=False: 仅标准 ChatML，模型自行生成完整 CoT + 答案

    注意: 必须手工拼 ChatML 并严格复刻 data.py 的训练格式（含 system message、
    vision 块与问题之间的换行）。Qwen3-VL 的 chat template 不会自动补默认
    system message（Qwen2.5 会），用 apply_chat_template 会造成训练/推理
    分布不一致，采样温度下生成直接崩坏。

    CoT 前缀复用 data.py 的 build_doc_cot：自适应预算下每张图的 token 数与出现
    的步骤都不同，写死 4/4/4 会造成同一类分布不一致；复用同一构造函数还能避免
    训练侧与 RL 侧格式漂移，并自动兼容索引槽位 token。
    """
    fixed_cot = build_doc_cot(get_anchor_budget(image_file)) + "\n<answer>"
    text_prompt = (
        f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}user\n"
        f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}\n"
        f"{question}\n{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}assistant\n"
    )
    if force_cot_prefix:
        return text_prompt + fixed_cot
    else:
        return text_prompt


class GRPOTrainer:
    def __init__(self, policy_model, ref_model, processor, config):
        """
        Args:
            policy_model: 当前策略模型 (可训练)
            ref_model: 参考模型 (冻结, 通常是 SFT checkpoint)
                        传 None 则使用 LoRA disable 技巧 (推荐,省显存)
            processor: AutoProcessor
            config: 包含 G, lr, beta, temperature, max_new_tokens 等
        """
        self.policy = policy_model
        self.ref = ref_model
        self.processor = processor
        self.config = config

        # 冻结参考模型 (如果提供了单独的 ref 模型)
        if self.ref is not None:
            for p in self.ref.parameters():
                p.requires_grad = False
            self.ref.eval()
            print("[GRPO] Using separate reference model")
        else:
            print("[GRPO] Using LoRA disable trick for reference (no separate ref model)")

        # 优化器只更新可训练参数 (优先使用 8-bit 优化器节省显存)
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=config.get('lr', 1e-6),
                weight_decay=0.0,
            )
            print("[GRPO] Using 8-bit AdamW optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.get('lr', 1e-6),
                weight_decay=0.0,
            )
            print("[GRPO] Using standard AdamW (install bitsandbytes for 8-bit)")

        self.device = next(self.policy.parameters()).device
        self.step_count = 0

        # 视觉对齐辅助 loss 权重
        self.visual_align_weight = config.get('visual_align_weight', 0.1)

        # CoT 前缀注入开关: True=注入前缀+跳过R_format, False=模型自生成CoT+计算R_format
        self.force_cot_prefix = config.get('force_cot_prefix', True)

        # 零方差跳过 (DAPO 式动态采样): 组内奖励无差异时 advantage 全零,
        # GRPO 无学习信号, 跳过昂贵的 backward 把算力留给有信号的样本
        self.skip_zero_variance = config.get('skip_zero_variance', True)
        self.zero_std_threshold = config.get('zero_std_threshold', 1e-6)
        self.skipped_count = 0

        # === GRPO v3 改进 (默认开启, 可用 config 回退到旧行为) ===
        # #1 答案段梯度: 只在 <answer> 之后的 token 上算策略梯度,
        #    屏蔽 force_cot_prefix=False 时模型自生成的固定 CoT 模板 token (梯度污染)
        self.answer_only_grad = config.get('answer_only_grad', True)
        # 自适应预算下 CoT 不再是常量（它编码"这份文档花多少视觉 token"），
        # 屏蔽 think 段等于禁止 RL 优化预算；答案项仍由 advantage 主导
        self.grad_scope = config.get('grad_scope', 'answer')   # 'answer' | 'full'
        # 视觉 token 开销权重：>0 时组内"全对但 token 数不同"才有奖励方差
        self.w_token = config.get('w_token', 0.0)
        self.token_cap = config.get('token_cap', 18)
        # 'mixed'=仅训跨界组; 'drop_allwrong'=保留全对组（学"该省"的唯一来源）; 'off'=不门控
        self.gate_mode = config.get('gate_mode', 'mixed')
        # 变长 CoT 的语法约束（标签顺序 / token 归位 / 分支容量），见 rl_reward.reward_cot_format
        self.w_cot_fmt = config.get('w_cot_fmt', 0.0)
        # 视觉对齐残差入奖励。只作辅助 loss 时它只训投影头，管不到"发多少 token / 省哪一步"；
        # 必须经 advantage 才能约束预算决策。与 w_token 反向，两者均衡点即真实预算。
        # >0 时每步额外一轮 no_grad forward（G 次），仅"自己发现预算"臂需要。
        self.w_align = config.get('w_align', 0.0)
        # <answer> 从 tokenizer 读取 (不硬编码, 兼容新结构模型)
        self.answer_token_id = self.processor.tokenizer.convert_tokens_to_ids('<answer>')
        # #2 优势归一化: 'mean'=Dr.GRPO 只去均值 (梯度量级∝真实质量差),
        #    'std'=旧版组内标准化 (抹平量级)
        self.adv_norm = config.get('adv_norm', 'mean')
        # #3 混合成功门控: 仅训"跨越正确性边界"(既有≥阈值又有<阈值)的组,
        #    全对/全错组跳过 (全错组会训成"错得更像GT措辞", 有害)
        self.mixed_gate = config.get('mixed_gate', True)
        self.success_threshold = config.get('success_threshold', 0.9)
        # #4 KL 估计: 'k3'=exp(Δ)-Δ-1 (非负低方差), 'k1'=旧版 logπ-logπ_ref (带符号高方差)
        self.kl_estimator = config.get('kl_estimator', 'k3')
        self.gated_count = 0
        print(f"[GRPO v3] answer_only_grad={self.answer_only_grad}(ans_id={self.answer_token_id}), "
              f"adv_norm={self.adv_norm}, mixed_gate={self.mixed_gate}(thr={self.success_threshold}), "
              f"kl={self.kl_estimator}")

        # 周期性 checkpoint (在 train_epoch 内按有效步触发;
        # 旧版在 epoch 层检查, 单 epoch 内永远不会保存)
        self.save_steps = config.get('save_steps', 0)
        self.output_dir = config.get('output_dir', None)

        # === teacher 模型保留在 GPU 0 (与 policy 同卡) ===
        # 不再 offload 到 GPU 1,省去跨卡数据传输开销
        # GPU 1 完全释放,可供后续并行优化使用
        self._keep_teacher_on_policy_device()

    def _keep_teacher_on_policy_device(self):
        """Teacher 模型保留在 policy 所在设备,不做跨卡 offload"""
        base = self.policy.get_base_model() if hasattr(self.policy, 'get_base_model') else self.policy
        anchor = getattr(base, 'anchor_models', None)
        if anchor is None:
            return
        print(f"[GRPO] Teacher models stay on {self.device} (same as policy, no offload)")

    def generate_group(self, image, question):
        """
        对单个 (image, question) 批量采样 G 条回答
        一次性生成 G 条轨迹,并截断 padding
        """
        G = self.config.get('G', 8)
        temperature = self.config.get('temperature', 0.7)
        max_new_tokens = self.config.get('max_new_tokens', 128)

        prompt = build_rl_prompt(question, self.processor, self.force_cot_prefix,
                                 image_file=image)
        print(f"[DEBUG] prompt: \n{prompt}")

        # 处理单条输入 (所有轨迹共享同一张图片)
        single_inputs = self.processor(
            text=[prompt], images=[image],
            padding=True, return_tensors='pt'
        ).to(self.device)

        prompt_len = single_inputs['input_ids'].shape[1]

        # 构建 batch 输入: 单条输入在第 0 维重复 G 次
        batch_inputs = {}
        for k, v in single_inputs.items():
            batch_inputs[k] = v.repeat(G, *([1] * (v.dim() - 1)))

        # 一次性生成 G 条轨迹
        # transformers 5.x: 开着 gradient checkpointing 时 generate 输出会崩坏
        # (乱码/混合语言), 生成期间必须临时关闭, backward 前再恢复
        base = self.policy.get_base_model() if hasattr(self.policy, 'get_base_model') else self.policy
        gc_was_on = getattr(base, 'is_gradient_checkpointing', False)
        was_training = self.policy.training
        if gc_was_on:
            base.gradient_checkpointing_disable()
        self.policy.eval()
        try:
            with torch.no_grad():
                output_ids = self.policy.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
        finally:
            if gc_was_on:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            if was_training:
                self.policy.train()

        # 获取 EOS/PAD token id 用于截断 padding
        eos_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id or eos_id

        results = []
        for i in range(G):
            # 截断 padding: 找到生成部分第一个 EOS/PAD 位置
            gen_tokens = output_ids[i, prompt_len:]
            # 找到第一个结束标记 (EOS 或 PAD)
            end_mask = (gen_tokens == eos_id) | (gen_tokens == pad_id)
            end_positions = end_mask.nonzero()
            if len(end_positions) > 0:
                gen_len = end_positions[0].item() + 1  # 包含 EOS
            else:
                gen_len = len(gen_tokens)  # 无 EOS,用全长

            # 截断后的 full_ids (不含 padding)
            actual_len = prompt_len + gen_len
            full_ids = output_ids[i:i+1, :actual_len]

            gen_text = self.processor.batch_decode(
                full_ids[:, prompt_len:],
                skip_special_tokens=True
            )[0].strip()

            results.append({
                'text': gen_text,
                'full_ids': full_ids,
                'prompt_len': prompt_len,
                'inputs': single_inputs,
                'image_files': [[image]],
            })

        return results

    def _visual_losses(self, trajectories):
        """每条轨迹的视觉对齐残差（no_grad，纯测量）。

        模型侧必须打开 charge_absent_branches：否则被整步省略的分支残差恒为 0，
        "全省"在奖励里免费，预算必然塌缩。
        """
        out = []
        for traj in trajectories:
            _, vl = self.compute_log_probs(
                self.policy, traj['full_ids'], traj['prompt_len'], traj['inputs'],
                image_files=traj.get('image_files'), no_grad=True)
            out.append(float(vl) if isinstance(vl, torch.Tensor) else 0.0)
        return out

    def compute_rewards(self, trajectories, gt_answer, visual_losses=None):
        """计算每条轨迹的奖励"""
        # 强制 CoT 前缀时生成文本不含 <answer> 标签，R_format 无效，跳过
        w_format = 0.0 if self.force_cot_prefix else 0.1
        rewards = []
        details = []
        for i, traj in enumerate(trajectories):
            r, d = compute_reward_with_cost(
                traj['text'], gt_answer, w_format=w_format,
                w_token=self.w_token, token_cap=self.token_cap,
                w_cot_fmt=self.w_cot_fmt, w_align=self.w_align,
                visual_loss=visual_losses[i] if visual_losses else None)
            rewards.append(r)
            details.append(d)
            print(f"[DEBUG] traj {i}: gen_text='{traj['text']}'")
            print(f"[DEBUG] traj {i}: pred_answer='{d['pred_answer']}'")
            print(f"[DEBUG] traj {i}: gt_answer='{gt_answer}'")
            print(f"[DEBUG] traj {i}: r_answer={d['r_answer']}, r_format={d['r_format']}, "
                  f"r_cot_fmt={d['r_cot_fmt']}, budget={d['budget']}, "
                  f"va={d['visual_loss']}, reward={r}")
        return torch.tensor(rewards, dtype=torch.float32), details

    def compute_advantages(self, rewards):
        """组内优势。adv_norm='mean' 只去均值(Dr.GRPO, 保留质量差量级);
        'std' 为旧版标准化 (会抹平 0.9/1.0 措辞差与 0.1/1.0 对错差的量级区别)"""
        adv = rewards - rewards.mean()
        if self.adv_norm == 'std':
            adv = adv / (rewards.std() + 1e-8)
        return adv

    def compute_log_probs(self, model, full_ids, prompt_len, inputs, image_files=None,
                          no_grad=False):
        """
        用 CoVT 原始 forward 计算逐 token log_prob 和纯视觉对齐 loss
        Returns: (answer_token_log_probs [T], visual_loss)
            answer_token_log_probs: answer_only_grad=True 时仅 <answer> 之后的 token;
                                    否则为全部生成 token。供 k3 KL 与策略梯度共用
            visual_loss: 纯视觉对齐 loss (不含 text_loss, 未乘权重); 无锚点 loss 时为 None
            no_grad: 只取数值不建图（对齐残差入奖励的前置测量用）。内部原先硬写
                     enable_grad，会覆盖外层 no_grad 上下文而白占显存
        """
        model_device = next(model.parameters()).device
        full_ids_dev = full_ids.to(model_device)

        # 掩码 prompt 部分，只在 generated token 上计算 text_loss
        labels = full_ids_dev.clone()
        labels[:, :prompt_len] = -100  # IGNORE_INDEX

        forward_kwargs = {
            'input_ids': full_ids_dev,
            'attention_mask': torch.ones_like(full_ids_dev),
            'labels': labels,
        }
        for key in ['pixel_values', 'image_grid_thw']:
            if key in inputs:
                forward_kwargs[key] = inputs[key].to(model_device)

        # 只在 anchor_models 可用时传 image_files (视觉对齐需要 teacher 模型)
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        if image_files is not None and getattr(base, 'anchor_models', None) is not None:
            forward_kwargs['image_files'] = image_files

        with (torch.no_grad() if no_grad else torch.enable_grad()):
            outputs = model(**forward_kwargs)
            logits = outputs.logits

            # 生成部分的 logits (shift by 1)
            gen_logits = logits[0, prompt_len - 1:-1, :]  # [gen_len, vocab]
            gen_tokens = full_ids[0, prompt_len:].to(model_device)  # [gen_len]

            log_probs = F.log_softmax(gen_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [gen_len]

            # #1 答案段定位: force_cot_prefix=False 时模型自生成 <think>..</think><answer>,
            # 取最后一个 <answer> 之后为答案段; force_cot_prefix=True 时 <answer> 在 prompt 尾,
            # gen_tokens 中无此 token, ans_start=0 (全部生成即答案), 两种模式都正确
            if self.answer_only_grad:
                if self.grad_scope != 'full':
                    pos = (gen_tokens == self.answer_token_id).nonzero()
                    ans_start = pos[-1].item() + 1 if len(pos) > 0 else 0
                    token_log_probs = token_log_probs[ans_start:]

            visual_loss = getattr(outputs, 'visual_loss', None)

        return token_log_probs, visual_loss

    def train_step(self, image, question, gt_answer):
        """
        单个样本的 GRPO 训练步骤 (含视觉对齐辅助 loss)
        """
        beta = self.config.get('beta', 0.04)
        va_weight = self.visual_align_weight

        # 1. 采样 G 条轨迹
        trajectories = self.generate_group(image, question)

        # 2. 计算奖励（w_align>0 时先测每条轨迹的对齐残差，让预算决策进入 advantage）
        visual_losses = self._visual_losses(trajectories) if self.w_align else None
        rewards, details = self.compute_rewards(trajectories, gt_answer, visual_losses)

        # 2.5 样本门控: 决定本组是否值得 backward
        if self.gate_mode != 'off' and self.mixed_gate:
            # 混合成功门控: 仅训"跨越正确性边界"的组
            # (既有≥阈值又有<阈值), 全对/全错组无可学习对比, 跳过
            # gate_mode='drop_allwrong': 只丢全错组, 保留全对组 ——
            # "全对但 token 数不同"是学"该省预算"的唯一来源, 有 w_token 时其奖励有方差
            hi = (rewards >= self.success_threshold)
            no_signal = rewards.std().item() < self.zero_std_threshold
            if self.gate_mode == 'drop_allwrong':
                # 保留全对组（token 开销使其奖励仍有方差），只丢全错组与无方差组
                drop = bool((~hi).all()) or no_signal
            else:
                drop = bool(hi.all() or (~hi).all())
            if drop:
                self.gated_count += 1
                kind = 'all-solved' if hi.all() else 'all-failed'
                print(f"[GRPO] Gate skip ({kind}, mean_r={rewards.mean().item():.3f}, "
                      f"total_gated={self.gated_count})")
                del trajectories
                torch.cuda.empty_cache()
                return None
        elif self.skip_zero_variance and rewards.std().item() < self.zero_std_threshold:
            # 旧版零方差跳过 (仅 mixed_gate=False 时启用)
            self.skipped_count += 1
            print(f"[GRPO] Skip zero-variance sample "
                  f"(reward={rewards.mean().item():.3f}, total_skipped={self.skipped_count})")
            del trajectories
            torch.cuda.empty_cache()
            return None

        # 3. 组内优势
        advantages = self.compute_advantages(rewards).to(self.device)

        torch.cuda.empty_cache()

        # 4. 逐条轨迹计算 log_prob + 视觉对齐 loss
        self.optimizer.zero_grad()
        total_loss_val = 0.0
        total_visual_val = 0.0
        G = len(trajectories)

        for i, traj in enumerate(trajectories):
            # 策略模型: 答案段逐 token log_prob (带梯度) + 纯视觉对齐 loss
            policy_lp, policy_visual_loss = self.compute_log_probs(
                self.policy, traj['full_ids'], traj['prompt_len'], traj['inputs'],
                image_files=traj.get('image_files'),
            )

            # 参考模型 log prob: LoRA disable 技巧 (逐条, 保证正确性)
            with torch.no_grad():
                if self.ref is not None:
                    ref_lp, _ = self.compute_log_probs(
                        self.ref, traj['full_ids'], traj['prompt_len'], traj['inputs'],
                        image_files=None, no_grad=True,
                    )
                else:
                    with self.policy.disable_adapter():
                        ref_lp, _ = self.compute_log_probs(
                            self.policy, traj['full_ids'], traj['prompt_len'], traj['inputs'],
                            image_files=None, no_grad=True,
                        )
                ref_lp = ref_lp.to(policy_lp.device).detach()

            # 答案段为空 (如生成仅 <answer> 无内容) 则跳过此轨迹
            T = min(policy_lp.shape[0], ref_lp.shape[0])
            if T == 0:
                del policy_lp, ref_lp
                continue
            policy_lp, ref_lp = policy_lp[:T], ref_lp[:T]
            policy_logp_sum = policy_lp.sum()

            # #4 KL 项: k3=exp(Δ)-Δ-1 (Δ=ref-policy, 非负低方差, token 级求和);
            # k1=logπ-logπ_ref (旧版, 带符号)
            if self.kl_estimator == 'k3':
                diff = ref_lp - policy_lp
                kl = (torch.exp(diff) - diff - 1).sum()
            else:
                kl = policy_logp_sum - ref_lp.sum()

            # GRPO loss + 纯视觉对齐辅助 loss (不含 text_loss)
            grpo_loss = (-advantages[i] * policy_logp_sum + beta * kl) / G
            va_loss = va_weight * policy_visual_loss / G if isinstance(policy_visual_loss, torch.Tensor) else 0.0

            loss = grpo_loss + va_loss
            loss.backward()
            total_loss_val += grpo_loss.item()
            if isinstance(va_loss, torch.Tensor):
                total_visual_val += va_loss.item()

            del policy_lp, ref_lp, policy_logp_sum, loss
            torch.cuda.empty_cache()

        # 5. 更新参数
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            max_norm=1.0
        )
        self.optimizer.step()
        self.step_count += 1

        return {
            'loss': total_loss_val,
            'visual_loss': total_visual_val,
            'mean_reward': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
            # 预算轨迹：RL 是否真在调预算只能从这里看，没有它无法区分"学到了"与"没动"
            'mean_tokens': sum(d['n_visual'] for d in details) / len(details),
            'token_spread': max(d['n_visual'] for d in details) - min(d['n_visual'] for d in details),
            'branch_tokens': {b: sum(d['budget'][b] for d in details) / len(details)
                              for b in details[0]['budget']},
            'mean_cot_fmt': sum(d['r_cot_fmt'] for d in details) / len(details),
            'step': self.step_count,
            'details': details,
        }

    def save_checkpoint(self, tag):
        """
        保存 checkpoint: PEFT adapter + processor + 非 LoRA 可训练参数
        注意: embed_tokens/lm_head/anchor projection 未注册进 modules_to_save,
        save_pretrained 只存 LoRA 权重, 必须额外保存这些参数否则更新丢失
        """
        if not self.output_dir:
            return
        path = os.path.join(self.output_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.processor.save_pretrained(path)
        extra = {n: p.detach().cpu() for n, p in self.policy.named_parameters()
                 if p.requires_grad and 'lora_' not in n}
        if extra:
            torch.save(extra, os.path.join(path, 'extra_trainable.bin'))
        print(f"[GRPO] Checkpoint saved: {path} "
              f"(adapter + {len(extra)} extra trainable tensors)")

    def train_epoch(self, dataloader, max_steps=None):
        """训练一个 epoch"""
        self.policy.train()
        logs = []

        for batch_idx, batch in enumerate(dataloader):
            # 按有效步数(完成参数更新的步)停止, 被跳过的零方差样本不消耗预算
            if max_steps and len(logs) >= max_steps:
                break

            raw_image = Image.open(batch['image_path'][0]).convert('RGB')
            # 与 eval 脚本一致：只在图片过大时缩小，不强制 resize
            w, h = raw_image.size
            if w * h > 1400000:
                scale = (1400000 / (w * h)) ** 0.5
                image = raw_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            else:
                image = raw_image
            question = batch['question'][0]
            gt_answer = batch['gt_answer'][0]

            try:
                log = self.train_step(image, question, gt_answer)
                if log is None:
                    # 零方差样本被跳过, 不计入有效步数
                    continue
                logs.append(log)

                if self.step_count % 1 == 0:
                    bt = log['branch_tokens']
                    print(f"[Step {log['step']}] "
                          f"loss={log['loss']:.4f} "
                          f"va={log.get('visual_loss', 0):.4f} "
                          f"reward={log['mean_reward']:.3f}±{log['reward_std']:.3f} "
                          f"[{log['min_reward']:.2f},{log['max_reward']:.2f}] "
                          f"tok={log['mean_tokens']:.2f}(spread{log['token_spread']}) "
                          f"det/lay/flow={bt.get('det', 0):.1f}/{bt.get('layout', 0):.1f}"
                          f"/{bt.get('flow', 0):.1f} fmt={log['mean_cot_fmt']:.2f}")

                # 周期性 checkpoint (按有效步计数)
                if self.save_steps and self.step_count % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.step_count}")

            except Exception as e:
                print(f"[Error] Step failed: {e}")
                continue

        return logs
