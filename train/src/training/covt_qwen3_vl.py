"""Doc-SCoT on Qwen3-VL: 薄包装方案。

直接继承 HF transformers 5.x 的 Qwen3VLForConditionalGeneration，图像注入 / deepstack /
interleaved mrope / KV cache / prepare_inputs_for_generation 全部复用父类实现，
本文件只负责 Doc-SCoT 特有部分：det/layout/flow 视觉思维 token 的提取、解码与 anchor loss。

teacher 侧（AnchorModels/AnchorLoss）与底座无关，见 training/anchor_teachers.py。
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import wandb

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from training.anchor_teachers import AnchorModels, AnchorLoss


@dataclass
class CoVTQwen3VLCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    anchor_outputs: Optional[Tuple] = None
    # 纯视觉对齐 loss（未乘 visual_weight，不含 text_loss）；全部锚点 loss 为 0 时为 None
    visual_loss: Optional[torch.FloatTensor] = None


class CoVTQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.text_config.hidden_size

        self.anchor_model_id = None
        self.anchor_loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # AnchorLoss 签名兼容
        self.anchor_models = None
        self.anchor_loss = None

        # 单位：优化器步数。SFT 时由 StepSyncCallback 每步写入（external_step_control=True）；
        # 非 Trainer 场景（RL/推理）保留 forward 自增 fallback
        self.global_steps = 0
        self.external_step_control = False
        # liger 不支持 qwen3_vl（且 0.5.5 不兼容 transformers 5.x），FLCE 恒关，
        # 属性仅为 train.py 接口兼容保留
        self.use_flce_text_loss = False
        self.align_anchor_task_only_stage = 0
        self.total_training_steps = 6000
        self.linear_visual_weight_decay = True
        self.visual_weight_start = 0.2
        self.visual_weight_end = 0.0

        self.det_token_idx = None
        self.layout_token_idx = None
        self.flow_token_idx = None
        # 索引槽位模式下由 train.py 注入各分支 8 个槽位 token 的 id；None 表示相同 pad 模式
        self.det_slot_token_ids = None
        self.layout_slot_token_ids = None
        self.flow_slot_token_ids = None

        # 监督信号增强（默认关，旧脚本与旧 checkpoint 行为不变）；语义见 _branch_visual_loss
        self.prefix_supervision = False
        self.prefix_fracs = (0.5,)
        self.prefix_weight = 0.5
        self.charge_absent_branches = False

        # det: LLM 隐层 -> DBNet 特征空间 (256)
        self.det_projection = nn.Linear(hidden_size, 256)
        # query bank 固定 4 行，恒全部用于解码读出
        _qbank = 4
        self.det_query_vectors = nn.Parameter(torch.randn(_qbank, 256, dtype=torch.bfloat16))
        self.det_cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        # layout: LLM 隐层 -> DocLayout-YOLO 特征空间 (384)
        self.layout_projection = nn.Linear(hidden_size, 384)
        self.layout_query_vectors = nn.Parameter(torch.randn(_qbank, 384, dtype=torch.bfloat16))
        self.layout_cross_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)

        # flow: 解码特征源 = DBNet neck 128×128 (256ch) + 8 个坐标通道 = 264
        self.flow_projection = nn.Linear(hidden_size, 264)
        self.flow_query_vectors = nn.Parameter(torch.randn(_qbank, 264, dtype=torch.bfloat16))
        self.flow_cross_attention = nn.MultiheadAttention(embed_dim=264, num_heads=8, batch_first=True)

    def _init_weights(self, module):
        # transformers 5.x 的 from_pretrained 走 meta-device 初始化，__init__ 中的
        # randn/MHA 默认值会被丢弃，缺失权重仅经 _init_weights 物化；基类不认识
        # nn.MultiheadAttention 与裸 nn.Parameter，会留下未初始化内存（随机出现 NaN），
        # 必须在此显式初始化
        super()._init_weights(module)
        if isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif module is self:
            for name in ("det_query_vectors", "layout_query_vectors", "flow_query_vectors"):
                param = getattr(self, name, None)
                if param is not None:
                    param.data.normal_(mean=0.0, std=1.0)

    def get_anchor_model_ids(self, anchor_model_id):
        self.anchor_model_id = anchor_model_id
        self.anchor_models = AnchorModels(self.anchor_model_id)
        self.anchor_loss = AnchorLoss(self.anchor_loss_weight)

    def get_anchor_token_idx(self, det_token_idx=None, layout_token_idx=None,
                             flow_token_idx=None):
        self.det_token_idx = det_token_idx
        self.layout_token_idx = layout_token_idx
        self.flow_token_idx = flow_token_idx

    def _anchor_mask(self, input_ids, branch):
        """该分支 anchor token 的位置掩码。

        索引槽位模式下一个分支有 8 个不同的槽位 token，需按集合匹配；
        未注入槽位 id 时退回单 pad id 比较（相同 token 模式）。
        """
        slot_ids = getattr(self, f"{branch}_slot_token_ids", None)
        if slot_ids is not None:
            return torch.isin(input_ids, slot_ids.to(input_ids.device))
        return input_ids == getattr(self, f"{branch}_token_idx")

    def _gather_anchor_hidden(self, last_hidden_state, mask):
        """收集各样本 anchor 位置的隐层。

        自适应预算下每个样本的 anchor token 数不同，无法直接 stack，
        因此右侧 padding 到 batch 内最大长度，并返回 key_padding_mask
        （True = padding 位；cross-attention 据此忽略这些位置）。
        返回 (padded[B',L,H], key_padding_mask[B',L], valid_indices)；无有效样本时返回 (None, None, [])。
        """
        feats, valid = [], []
        for i in range(mask.shape[0]):
            if mask[i].any():
                valid.append(i)
                feats.append(last_hidden_state[i, mask[i]])
        if not feats:
            return None, None, []
        max_len = max(f.shape[0] for f in feats)
        padded = last_hidden_state.new_zeros(len(feats), max_len, feats[0].shape[-1])
        kpm = torch.ones(len(feats), max_len, dtype=torch.bool, device=last_hidden_state.device)
        for i, f in enumerate(feats):
            padded[i, :f.shape[0]] = f
            kpm[i, :f.shape[0]] = False
        return padded, kpm, valid

    # 各分支解码特征场的空间尺寸；flow 额外拼 8 个坐标通道
    _BRANCH_SPEC = {
        'det': dict(target_hw=(128, 128), add_coords=False),
        'layout': dict(target_hw=(32, 32), add_coords=False),
        'flow': dict(target_hw=(128, 128), add_coords=True),
    }

    def _prefix_variants(self):
        """本步要监督的 (前缀比例, 权重)，恒含全长 (1.0, 1.0)。"""
        if not getattr(self, 'prefix_supervision', False):
            return ((1.0, 1.0),)
        w = getattr(self, 'prefix_weight', 0.5)
        return ((1.0, 1.0),) + tuple((f, w) for f in getattr(self, 'prefix_fracs', (0.5,)))

    @staticmethod
    def _truncate_kpm(kpm, frac):
        """只保留各样本前 ceil(frac*k) 个 anchor token（至少 1 个）。"""
        counts = (~kpm).sum(dim=1)
        lim = torch.ceil(counts.to(torch.float32) * frac).clamp(min=1).long()
        ar = torch.arange(kpm.shape[1], device=kpm.device).unsqueeze(0)
        return kpm | (ar >= lim.unsqueeze(1))

    @staticmethod
    def _fill_pad_with_mean(x, kpm):
        """把 padding 位替换为该样本有效位的均值。

        无效位经此填充后，解码时对 token 维
        求均值的结果与"只对有效位求均值"完全一致。
        """
        valid = (~kpm).unsqueeze(-1).to(x.dtype)
        mean = (x * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.where(kpm.unsqueeze(-1), mean.expand_as(x), x)

    def _zero_touch_anchor_params(self):
        """返回数值恒为 0、但与三个分支全部参数相连的项。

        自适应预算下某分支可能在本 rank 的 micro-batch 里整步缺席（k=0），
        该分支参数在本 rank 拿不到梯度而在其它 rank 拿得到，DeepSpeed 的梯度
        归约两侧集合操作不一致，会直接 NCCL ALLREDUCE 超时死锁。加上这一项可
        保证各 rank 参与归约的参数集合始终一致（梯度值仍为 0，不影响优化）。
        """
        total = None
        mods = (self.det_projection, self.det_cross_attention,
                self.layout_projection, self.layout_cross_attention,
                self.flow_projection, self.flow_cross_attention)
        params = [p for mod in mods for p in mod.parameters()]
        params += [getattr(self, n) for n in
                   ("det_query_vectors", "layout_query_vectors", "flow_query_vectors")
                   if getattr(self, n, None) is not None]
        for p in params:
            if not p.requires_grad:
                continue
            t = p.sum() * 0.0
            total = t if total is None else total + t
        return total

    def _branch_readout(self, branch, hidden, kpm, out_dtype):
        """一个分支在给定 kpm 下的解码权重读出（恒用全部 4 行 query bank）。
        kpm 被截断时即前缀读出。"""
        proj = getattr(self, f'{branch}_projection')
        bank = getattr(self, f'{branch}_query_vectors')
        kv = nn.functional.normalize(proj(hidden), dim=-1)
        query = bank.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        out, _ = getattr(self, f'{branch}_cross_attention')(
            query=query.to(out_dtype), key=kv, value=kv, key_padding_mask=kpm)
        return out

    def _prior_readout(self, branch, n, out_dtype):
        """k=0 时的先验读出：固定 query bank 直接作解码权重（图像无关探针）。"""
        bank = getattr(self, f'{branch}_query_vectors')
        return bank.unsqueeze(0).expand(n, -1, -1).to(out_dtype)

    def _branch_visual_loss(self, branch, last_hidden_state, input_ids, image_grid_thw,
                            encoded_value, image_files, per_sample_loss):
        """一个分支的视觉对齐 loss。

        除"k 个 token 池化后重建整图"的原监督外，含两项让预算本身产生意义的信号：

        1) 前缀嵌套监督：额外用前 j 个 token 再解码一次（j=ceil(frac*k)，
           解码时 query bank 恒取全部）。第 1 个 token 必须独立成粗摘要、
           后续 token 只能补增量，token 之间因此被迫分工而非互换；loss 随 k 单调下降后，
           L(k) 的拐点才是这张图真实需要的预算，不必再依赖离线启发式。

        2) 缺席分支先验计价：k=0 时改用 _prior_readout 解码，其残差即"整步省略丢掉多少
           信息"——平凡版面固定探针就能解释、省得便宜；复杂版面残差大、省得贵。不加这项时
           被省略分支的 loss 恒为 0，"全部省掉"是免费最优解，且 RL 侧无法给省略定价。
        """
        dtype = last_hidden_state.dtype
        spec = self._BRANCH_SPEC[branch]
        hidden, kpm, valid = self._gather_anchor_hidden(
            last_hidden_state, self._anchor_mask(input_ids, branch))

        def feats_of(indices):
            return encoded_value[indices].to(dtype)

        def loss_of(readout, indices):
            pred = self.anchor_models.decode_det_tokens(readout, feats_of(indices))
            total = 0.0
            for idx, b in enumerate(indices):
                total = total + per_sample_loss(pred[idx:idx + 1], image_files[b][0], dtype)
            return total / len(indices)

        branch_loss = 0.0
        if hidden is not None:
            acc, wsum = 0.0, 0.0
            for frac, w in self._prefix_variants():
                kpm_v = kpm if frac >= 1.0 else self._truncate_kpm(kpm, frac)
                acc = acc + w * loss_of(self._branch_readout(branch, hidden, kpm_v, dtype), valid)
                wsum += w
            branch_loss = acc / wsum

        if getattr(self, 'charge_absent_branches', False):
            present = set(valid)
            absent = [i for i in range(input_ids.shape[0]) if i not in present]
            if absent:
                prior = loss_of(self._prior_readout(branch, len(absent), dtype), absent)
                n = input_ids.shape[0]
                branch_loss = (branch_loss * len(present) + prior * len(absent)) / n
        return branch_loss

    def _flow_sample_loss(self, pred, image_file, dtype):
        """flow 监督为剪影 GT + 按阅读顺序排列的框（ranking），签名与另两支不同。"""
        sil_gt, boxes_in_order = self.anchor_models.get_flow_supervision(image_file)
        return self.anchor_loss.get_flow_loss(pred, sil_gt.to(dtype), boxes_in_order)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_files=None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        second_per_grid_ts=None,  # 旧 data.py 视频路径会传；Qwen3-VL 用 timestamp token，不接受该参数
        logits_to_keep=0,
        **kwargs,
    ):
        if not self.external_step_control:
            self.global_steps += 1

        if self.anchor_models is not None and not getattr(self.anchor_models, '_runtime_configured', False):
            # teacher 冻结且设备固定，只需配置一次
            self.anchor_models.set_device(self.device)
            self.anchor_models.set_float()
            self.anchor_models._runtime_configured = True

        # Qwen3-VL 的 mrope 必需 mm_token_type_ids（text=0/image=1/video=2）；
        # 训练 collator 若未提供则从 input_ids 推导（image/video pad token 位置即视觉段）
        if (mm_token_type_ids is None and input_ids is not None
                and (image_grid_thw is not None or video_grid_thw is not None)):
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            mm_token_type_ids[input_ids == self.config.video_token_id] = 2

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        predict_det_dict = None
        predict_layout_dict = None
        predict_flow_dict = None

        det_loss = 0.0
        layout_loss = 0.0
        flow_loss = 0.0

        det_encoded_value = None
        layout_encoded_value = None
        flow_encoded_value = None

        if image_files and self.global_steps <= self.total_training_steps:
            det_encoded_values = []
            layout_encoded_values = []
            flow_encoded_values = []
            for image_file in image_files:
                det_encoded_values.append(self.anchor_models.get_det_embed(image_file[0]))
                layout_encoded_values.append(self.anchor_models.get_layout_embed(image_file[0]))
                flow_encoded_values.append(self.anchor_models.get_flow_embed(image_file[0]))
            det_encoded_value = torch.cat(det_encoded_values, dim=0).to(last_hidden_state.dtype) if (det_encoded_values[0] is not None) else None
            layout_encoded_value = torch.cat(layout_encoded_values, dim=0).to(last_hidden_state.dtype) if (layout_encoded_values[0] is not None) else None
            flow_encoded_value = torch.cat(flow_encoded_values, dim=0).to(last_hidden_state.dtype) if (flow_encoded_values[0] is not None) else None

        # === 三分支视觉对齐 loss ===
        # 三分支结构相同，仅解码特征场与 GT/loss 不同；前缀嵌套监督与缺席分支计价
        # 统一在 _branch_visual_loss 内实现，避免三处拷贝导致行为漂移。
        branch_losses = {}
        for branch, encoded_value, per_sample_loss in (
            ('det', det_encoded_value,
             lambda pred, f, dt: self.anchor_loss.get_det_loss(
                 pred, self.anchor_models.get_det_gt(f).to(dt))),
            ('layout', layout_encoded_value,
             lambda pred, f, dt: self.anchor_loss.get_layout_loss(
                 pred, self.anchor_models.get_layout_gt(f).to(dt))),
            ('flow', flow_encoded_value, self._flow_sample_loss),
        ):
            if encoded_value is None:
                branch_losses[branch] = 0.0
                continue
            branch_losses[branch] = self._branch_visual_loss(
                branch, last_hidden_state, input_ids, image_grid_thw,
                encoded_value, image_files, per_sample_loss)
            if self.global_steps % 50 == 0:
                try:
                    wandb.log({f"{branch}_loss": branch_losses[branch]})
                except Exception:
                    pass
        det_loss = branch_losses['det']
        layout_loss = branch_losses['layout']
        flow_loss = branch_losses['flow']

        total_visual_loss = det_loss + layout_loss + flow_loss
        if self.training and image_files:
            # 自适应预算下分支可能整步缺席，需保证各 rank 参与梯度归约的参数集合一致
            zero_touch = self._zero_touch_anchor_params()
            if zero_touch is not None:
                total_visual_loss = total_visual_loss + zero_touch
        if self.global_steps % 50 == 0 and (det_encoded_value is not None or layout_encoded_value is not None or flow_encoded_value is not None):
            print(f'det_loss: {det_loss}, layout_loss: {layout_loss}, flow_loss: {flow_loss}, total_visual_loss: {total_visual_loss}')

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_hidden_state[:, slice_indices, :])

        text_loss = None
        if labels is not None:
            logits_f = logits.float()
            shift_logits = logits_f[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)
            if self.global_steps % 50 == 0:
                print(f'text_loss: {text_loss}')

        loss = None
        # === Stage 1: Visual Alignment ===
        if self.global_steps < self.align_anchor_task_only_stage:
            if labels is not None:
                loss = (text_loss * 0.2) + total_visual_loss
            else:
                loss = total_visual_loss
            if self.global_steps % 50 == 0:
                print(f'stage 1 loss: {loss}')
        # === Stage 2+: Joint Reasoning ===
        else:
            if labels is not None:
                if self.linear_visual_weight_decay:
                    # 衰减区间: Stage 2 的 1/5 处开始，4/5 处结束
                    stage2_start = self.align_anchor_task_only_stage
                    stage2_end = self.total_training_steps
                    stage2_len = stage2_end - stage2_start
                    decay_start = stage2_start + int(stage2_len * 0.2)
                    decay_end = stage2_start + int(stage2_len * 0.8)

                    if self.global_steps < decay_start:
                        visual_weight = self.visual_weight_start
                    elif self.global_steps >= decay_end:
                        visual_weight = self.visual_weight_end
                    else:
                        progress = (self.global_steps - decay_start) / max(decay_end - decay_start, 1)
                        visual_weight = self.visual_weight_start + (self.visual_weight_end - self.visual_weight_start) * progress
                else:
                    visual_weight = self.visual_weight_start
                loss = text_loss + (total_visual_loss * visual_weight)
                if self.global_steps % 50 == 0:
                    print(f'joint loss: {loss} (visual_weight={visual_weight:.4f})')

        anchor_outputs = (predict_det_dict, predict_layout_dict, predict_flow_dict)

        return CoVTQwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            anchor_outputs=anchor_outputs,
            visual_loss=total_visual_loss if isinstance(total_visual_loss, torch.Tensor) else None,
        )
