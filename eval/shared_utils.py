# shared_utils.py

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import math

# ========================================
#          Helper Classes
# ========================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FocusedSoftIntervener:
    """
    [v20 - 即时快照最终版]
    通过在猴子补丁内部，同时捕获干预前和干预后的注意力图，
    来获得最纯净、最精确的"那一瞬间"的变化。
    """
    def __init__(self, model, prior_mask, strength, image_feature_len, grid_reshape_size, attention_position="center", target_layers=None, bbox=None):
        self.model = model
        self.original_attention_function = None
        self.text_embeds_len = None
        self.strength = strength
        self.image_feature_len = image_feature_len
        self.attention_position = attention_position
        self.target_layers = target_layers if target_layers is not None else []
        self.current_layer = None
        self.bbox = bbox  # 新增bbox参数，用于指定注意力干预区域
        
        self.intervention_reward = self.prepare_prior_for_rewarding(prior_mask, grid_reshape_size)
        
        # [核心] 用于存储"每一时刻"的快照
        self.attention_snapshots = {} # key: (token_idx, layer_idx), value: {'before': tensor, 'after': tensor}
        self.is_intervening = True # 是否记录
        self.token_counter = 0  # 用于跟踪当前是第几个token

    def set_text_embeds_len(self, length: int):
        self.text_embeds_len = length


    def prepare_prior_for_rewarding(self, prior_mask_2d, grid_reshape_size):
        # print(f"DEBUG prepare_prior_for_rewarding: attention_position={self.attention_position}, grid_reshape_size={grid_reshape_size}")
        if self.attention_position != "custom" and self.attention_position != "bbox":
            h, w = grid_reshape_size
            reward_2d = np.zeros((h, w), dtype=np.float32)
            # print(f"DEBUG prepare_prior_for_rewarding: Creating {h}x{w} reward map")
            if self.attention_position == "center":
                center_h, center_w = h // 2, w // 2; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "top_left":
                center_h, center_w = h // 4, w // 4; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "top_right":
                center_h, center_w = h // 4, 3 * w // 4; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "bottom_left":
                center_h, center_w = 3 * h // 4, w // 4; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "bottom_right":
                center_h, center_w = 3 * h // 4, 3 * w // 4; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "horizontal_bar":
                center_row = h // 2; bar_height = max(1, h // 10)
                start_row = max(0, center_row - bar_height // 2); end_row = min(h, center_row + bar_height // 2)
                reward_2d[start_row:end_row, :] = 1.0
            elif self.attention_position == "vertical_bar":
                center_col = w // 2; bar_width = max(1, w // 10)
                start_col = max(0, center_col - bar_width // 2); end_col = min(w, center_col + bar_width // 2)
                reward_2d[:, start_col:end_col] = 1.0
            elif self.attention_position == "random":
                np.random.seed(42); reward_2d = np.random.rand(h, w)
            #  添加4个角点位置
            elif self.attention_position == "corner_top_left":
                center_h, center_w = 0, 0; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "corner_top_right":
                center_h, center_w = 0, w-1; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "corner_bottom_left":
                center_h, center_w = h-1, 0; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "corner_bottom_right":
                center_h, center_w = h-1, w-1; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            #  添加4个边的中点位置
            elif self.attention_position == "midpoint_top":
                center_h, center_w = 0, w // 2; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "midpoint_bottom":
                center_h, center_w = h-1, w // 2; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "midpoint_left":
                center_h, center_w = h // 2, 0; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            elif self.attention_position == "midpoint_right":
                center_h, center_w = h // 2, w-1; sigma = min(h, w) / 6
                for i in range(h):
                    for j in range(w): reward_2d[i, j] = np.exp(-((i - center_h)**2 + (j - center_w)**2) / (2 * sigma**2))
            reward_1d = torch.tensor(reward_2d.flatten(), dtype=torch.float32)
            # print(f"DEBUG prepare_prior_for_rewarding: Created reward map with max value {torch.max(reward_1d)} at index {torch.argmax(reward_1d)}")
        elif self.attention_position == "bbox" and self.bbox is not None:
            # 基于bbox创建注意力奖励图（支持多个bbox）
            h, w = grid_reshape_size
            reward_2d = np.zeros((h, w), dtype=np.float32)
            
            # 处理多个bbox
            bboxes = self.bbox if isinstance(self.bbox, list) else [self.bbox]
            
            for bbox in bboxes:
                # bbox格式: [x1, y1, x2, y2]，值在0-1之间
                x1, y1, x2, y2 = bbox
                
                # 将相对坐标转换为网格坐标
                x1_grid = int(x1 * w)
                y1_grid = int(y1 * h)
                x2_grid = int(x2 * w)
                y2_grid = int(y2 * h)
                
                # 确保坐标在有效范围内
                x1_grid = max(0, min(x1_grid, w-1))
                y1_grid = max(0, min(y1_grid, h-1))
                x2_grid = max(0, min(x2_grid, w-1))
                y2_grid = max(0, min(y2_grid, h-1))
                
                # 在bbox区域设置奖励值
                reward_2d[y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1.0
            
            reward_1d = torch.tensor(reward_2d.flatten(), dtype=torch.float32)
        else:
            if prior_mask_2d is None: return None
            h, w = grid_reshape_size
            prior_pil = Image.fromarray(prior_mask_2d.astype(np.uint8) * 255)
            resized_prior = prior_pil.resize((w, h), Image.Resampling.NEAREST)
            reward_1d = torch.tensor(np.array(resized_prior), dtype=torch.float32).flatten()
            reward_1d[reward_1d > 0] = 1.0
            # print(f"DEBUG prepare_prior_for_rewarding: Created custom reward map with {torch.sum(reward_1d > 0)} positive values")
        
        if len(reward_1d) != self.image_feature_len:
            # print(f"DEBUG prepare_prior_for_rewarding: Interpolating reward map from {len(reward_1d)} to {self.image_feature_len}")
            reward_1d = F.interpolate(reward_1d.view(1, 1, -1), size=self.image_feature_len, mode='nearest').view(-1)
        # print(f"DEBUG prepare_prior_for_rewarding: Final reward map shape: {reward_1d.shape}")
        return reward_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0)




    def _intervened_eager_attention_forward(
        self, module, query, key, value, attention_mask=None, **kwargs
    ):
        attn_output_orig, attn_weights_orig = self.original_attention_function(
            module, query, key, value, attention_mask=attention_mask, **kwargs
        )
        
        is_cross_attention = query.shape[-2] != key.shape[-2]
        layer_should_be_intervened = (not self.target_layers) or (self.current_layer in self.target_layers)
        
        # 添加调试信息
        # print(f"DEBUG: Layer {self.current_layer}, is_cross_attention: {is_cross_attention}, query.shape: {query.shape}, key.shape: {key.shape}")
        
        # 在生成模式下，我们需要通过其他方式跟踪token计数
        # 通过检查key的长度变化来判断是否是新的生成步骤
        if not hasattr(self, '_last_key_len'):
            self._last_key_len = key.shape[-2]
            # print(f"DEBUG: Initialized _last_key_len to {self._last_key_len}")
        
        # 如果key长度增加了，说明是新的生成步骤（新的token被添加到上下文中）
        # 但在自回归生成中，通常是逐个token生成的，所以我们需要另一种方式跟踪
        
        # 捕获和干预
        if self.is_intervening and is_cross_attention:
            # print(f"DEBUG: Intervening at layer {self.current_layer}, token {self.token_counter}")
            if layer_should_be_intervened and self.intervention_reward is not None and self.text_embeds_len is not None:
                key_states = repeat_kv(key, module.num_key_value_groups)
                image_start_index = self.text_embeds_len
                image_end_index = image_start_index + self.image_feature_len
                
                # print(f"DEBUG: text_embeds_len={self.text_embeds_len}, image_feature_len={self.image_feature_len}")
                # print(f"DEBUG: image_start_index={image_start_index}, image_end_index={image_end_index}")
                # print(f"DEBUG: key_states.shape={key_states.shape}")
                
                if key_states.shape[-2] >= image_end_index:
                    scaling = kwargs.get('scaling', module.scaling)
                    attn_scores_before_softmax = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                    if attention_mask is not None:
                         attn_scores_before_softmax += attention_mask[:, :, :, : key_states.shape[-2]]
                    
                    attn_scores_intervened = attn_scores_before_softmax.clone()
                    device_reward = self.intervention_reward.to(attn_scores_intervened.device)
                    # print(f"DEBUG: intervention_reward shape: {self.intervention_reward.shape}")
                    # print(f"DEBUG: Applying reward to indices {image_start_index}:{image_end_index}")
                    attn_scores_intervened[:, :, :, image_start_index:image_end_index] += (device_reward * self.strength)
                    
                    attn_weights_intervened = nn.functional.softmax(attn_scores_intervened, dim=-1, dtype=torch.float32).to(query.dtype)
                    dropout = kwargs.get('dropout', 0.0)
                    attn_weights_intervened = nn.functional.dropout(attn_weights_intervened, p=dropout, training=module.training)
                    value_states = repeat_kv(value, module.num_key_value_groups)
                    attn_output_intervened = torch.matmul(attn_weights_intervened, value_states)
                    attn_output_intervened = attn_output_intervened.transpose(1, 2).contiguous()
                    
                    # --- [核心] 捕获快照 ---
                    # 保存每个token、每层的注意力快照
                    # print(f"DEBUG: Saving snapshot for token {self.token_counter}, layer {self.current_layer}")
                    self.attention_snapshots[(self.token_counter, self.current_layer)] = {
                        'before': attn_weights_orig.detach().cpu(),
                        'after': attn_weights_intervened.detach().cpu()
                    }
                    
                    return attn_output_intervened, attn_weights_intervened
            
            # 即使不干预，也要捕获非目标层的原始状态作为'before'和'after'
            # 这样我们就能对比所有层
            # print(f"DEBUG: Saving non-intervened snapshot for token {self.token_counter}, layer {self.current_layer}")
            self.attention_snapshots[(self.token_counter, self.current_layer)] = {
                'before': attn_weights_orig.detach().cpu(),
                'after': attn_weights_orig.detach().cpu() # 因为没有干预，所以after=before
            }

        return attn_output_orig, attn_weights_orig

    def apply(self):
        try:
            import transformers
            modeling_module = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
            self.original_attention_function = modeling_module.eager_attention_forward
            
            self.layer_hooks = []
            import weakref
            self_ref = weakref.ref(self)
            
            def create_layer_hook(layer_idx):
                def hook(module, input, output):
                    if self_ref(): 
                        self_ref().current_layer = layer_idx
                        # 每当进入新的一层时，增加token计数器
                        # 这是因为在自回归生成中，每层处理完表示一个token生成完成
                        if layer_idx == 0 and hasattr(self_ref(), '_last_layer') and self_ref()._last_layer == 27:
                            self_ref().token_counter += 1
                            # print(f"DEBUG: Token counter increased to {self_ref().token_counter} after completing a full layer cycle")
                        self_ref()._last_layer = layer_idx
                return hook
            
            for i, layer in enumerate(self.model.language_model.layers):
                hook = layer.register_forward_hook(create_layer_hook(i))
                self.layer_hooks.append(hook)
            
            # 重置状态
            self.is_intervening = True
            self.attention_snapshots = {}
            self.token_counter = 0
            if hasattr(self, '_last_query_len'):
                delattr(self, '_last_query_len')
            if hasattr(self, '_last_key_len'):
                delattr(self, '_last_key_len')
            if hasattr(self, '_last_layer'):
                delattr(self, '_last_layer')
            
            def new_func(module, query, key, value, attention_mask, **kwargs):
                return self._intervened_eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
            modeling_module.eager_attention_forward = new_func
            # print("INFO: Focused Soft Intervener has been successfully applied.")
        except Exception as e:
            print(f"ERROR: Failed to apply Focused Soft Intervener: {e}"); self.remove()


    def remove(self):
        # 在移除后，将标志位置为False，这样同一个实例就不会再次捕获
        self.is_intervening = False
        self.token_counter = 0
        if hasattr(self, '_last_query_len'):
            delattr(self, '_last_query_len')
        if hasattr(self, '_last_key_len'):
            delattr(self, '_last_key_len')
        if hasattr(self, '_last_layer'):
            delattr(self, '_last_layer')
        
        if self.original_attention_function is not None:
            import transformers
            modeling_module = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
            modeling_module.eager_attention_forward = self.original_attention_function
            self.original_attention_function = None
            
            for hook in self.layer_hooks:
                hook.remove()
            self.layer_hooks = []
            
            # print("INFO: Focused Soft Intervener has been removed.")
    
    def get_snapshots(self):
        return self.attention_snapshots



class ControlGroupIntervener:
    """
    [对照组]
    一个只进行猴子补丁，但内部实现完全调用原始函数的对照干预器。
    目的是为了验证“猴子补丁”这一行为本身是否会引入与原始推理的差异。
    它不包含任何干预逻辑（strength, reward等），只负责替换、调用、捕获。
    """
    def __init__(self, model):
        self.model = model
        self.original_attention_function = None
        self.current_layer = None
        self.layer_hooks = []
        
        # 核心：只用于存储快照，不做任何干预
        self.attention_snapshots = {}
        self.is_capturing = True
        self.token_counter = 0

    def _control_eager_attention_forward(
        self, module, query, key, value, attention_mask=None, **kwargs
    ):
        # [核心区别] 不做任何自定义计算，直接调用并返回原始函数的结果
        attn_output, attn_weights = self.original_attention_function(
            module, query, key, value, attention_mask=attention_mask, **kwargs
        )
        
        is_cross_attention = query.shape[-2] != key.shape[-2]
        
        # [核心区别] 只在交叉注意力时捕获，不做任何干预判断
        if self.is_capturing and is_cross_attention:
            # 因为没有干预，所以 before 和 after 是完全一样的
            self.attention_snapshots[(self.token_counter, self.current_layer)] = {
                'before': attn_weights.detach().cpu(),
                'after': attn_weights.detach().cpu()
            }

        return attn_output, attn_weights

    def apply(self):
        try:
            import transformers
            modeling_module = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
            self.original_attention_function = modeling_module.eager_attention_forward
            
            # --- 与 FocusedSoftIntervener 完全相同的 Hook 设置 ---
            import weakref
            self_ref = weakref.ref(self)
            def create_layer_hook(layer_idx):
                def hook(module, input, output):
                    if self_ref(): 
                        self_ref().current_layer = layer_idx
                        if layer_idx == 0 and hasattr(self_ref(), '_last_layer') and self_ref()._last_layer == 27:
                            self_ref().token_counter += 1
                        self_ref()._last_layer = layer_idx
                return hook
            
            for i, layer in enumerate(self.model.language_model.layers):
                hook = layer.register_forward_hook(create_layer_hook(i))
                self.layer_hooks.append(hook)
            
            # --- 重置状态 ---
            self.is_capturing = True
            self.attention_snapshots = {}
            self.token_counter = 0
            if hasattr(self, '_last_layer'):
                delattr(self, '_last_layer')
            
            # --- 替换为本类的 control 函数 ---
            def new_func(module, query, key, value, attention_mask, **kwargs):
                return self._control_eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
            modeling_module.eager_attention_forward = new_func
            print("INFO: Control Group Intervener has been successfully applied.")
        except Exception as e:
            print(f"ERROR: Failed to apply Control Group Intervener: {e}"); self.remove()

    def remove(self):
        self.is_capturing = False
        if self.original_attention_function is not None:
            import transformers
            modeling_module = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
            modeling_module.eager_attention_forward = self.original_attention_function
            self.original_attention_function = None
            
            for hook in self.layer_hooks:
                hook.remove()
            self.layer_hooks = []
            
            print("INFO: Control Group Intervener has been removed.")
    
    def get_snapshots(self):
        return self.attention_snapshots



class CrossAttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.model.config.output_attentions = True
        self.model.language_model.config.output_attentions = True
        self.hooks = self._register_hooks()

    def _get_attention_module(self, layer):
        return layer.self_attn

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if output is not None and len(output) > 1 and output[1] is not None:
                self.attention_maps[layer_idx] = output[1].detach()
        return hook

    def _register_hooks(self):
        hooks = []
        for i, layer in enumerate(self.model.language_model.layers):
            module = self._get_attention_module(layer)
            hooks.append(module.register_forward_hook(self._hook_fn(i)))
        return hooks

    def remove_hooks(self):
        for hook in self.hooks: hook.remove()
        self.model.config.output_attentions = False
        self.model.language_model.config.output_attentions = False
        self.hooks = []

    # =========================================================================================
    #  请将 CrossAttentionVisualizer 类中的 process_and_visualize 方法完全替换为以下版本
    # =========================================================================================

    def process_and_visualize(self, full_inputs, text_embeds_len, answer_token_indices, image, save_path, image_grid_size, perform_visualization=True):
        """
        [最终正确版 - 带可视化开关]
        通过使用 F.interpolate 进行智能缩放来生成无扭曲的热力图。
        核心逻辑：
        1. 总是执行前向传播并计算热力图数据。
        2. 根据 perform_visualization 参数决定是否将热力图数据保存为图片文件。
        """
        self.attention_maps = {}
        self.model.eval()
        
        # 步骤 1: 总是执行前向传播以捕获所有注意力图
        with torch.no_grad():
            self.model(**full_inputs)

        if not self.attention_maps:
            if perform_visualization:
                print("错误：未能从任何层捕获到注意力图。")
            return None

        sample_attn_map = next(iter(self.attention_maps.values()))
        full_sequence_len = sample_attn_map.shape[-1]
        
        answer_len = len(answer_token_indices)
        resampled_image_len = full_sequence_len - text_embeds_len - answer_len
        
        rh, rw = image_grid_size
        target_len = rh * rw

        # 步骤 2: 总是计算平均热力图数据
        avg_heatmaps = {}
        for layer_idx, attention_map in sorted(self.attention_maps.items()):
            attention_map = attention_map.mean(dim=1).squeeze(0)
            
            image_start_index = text_embeds_len
            image_end_index = text_embeds_len + resampled_image_len
            
            if answer_token_indices:
                valid_indices = [idx for idx in answer_token_indices if idx < attention_map.shape[0]]
                if not valid_indices: continue

                img_attention_avg = attention_map[valid_indices, image_start_index:image_end_index].mean(dim=0)

                heatmap_avg = F.interpolate(
                    img_attention_avg.cpu().to(torch.float32).view(1, 1, -1), 
                    size=target_len, 
                    mode='linear', 
                    align_corners=False
                ).view(rh, rw).numpy()
                
                avg_heatmaps[layer_idx] = heatmap_avg
        
        # 步骤 3: 根据开关决定是否执行文件保存操作
        if perform_visualization:
            if avg_heatmaps:
                all_layers = sorted(avg_heatmaps.keys())
                last_10_layers = all_layers[-10:] if len(all_layers) >= 10 else all_layers
                selected_heatmaps = [avg_heatmaps[layer_idx] for layer_idx in last_10_layers]
                all_layers_avg_heatmap = np.mean(selected_heatmaps, axis=0)
                vis_img_all_layers = visualize_heatmap(all_layers_avg_heatmap, image)
                save_path_all_layers = os.path.join(save_path, "LAST_10_LAYERS_AVERAGE.jpg")
                Image.fromarray(vis_img_all_layers).save(save_path_all_layers)
                print(f"\n已保存后10层聚合的最终平均注意力图到: {save_path_all_layers}")
        
        # 步骤 4: 总是返回计算出的热力图数据
        return avg_heatmaps

    def get_attention_maps(self):
        """获取当前捕获的注意力图"""
        return self.attention_maps


# ========================================
#         Generic Helper Functions
# ========================================

def setup_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cudnn.benchmark = False; cudnn.deterministic = True

def visualize_heatmap(heatmap_raw, image):
    if heatmap_raw is None: return np.array(image)
    cam = cv2.resize(heatmap_raw, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
    cam -= np.min(cam)
    if np.max(cam) > 0: cam /= np.max(cam)
    
    cam = cv2.GaussianBlur(cam, (35, 35), 0)
    cam -= np.min(cam)
    if np.max(cam) > 0: cam /= np.max(cam)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * 0.5 + np.array(image) * 0.5
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def compare_attention_maps(normal_maps, intervened_maps, image, save_path):
    """比较正常和干预后的注意力图"""
    print("\n=== 注意力图比较 ===")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 确保两个字典有相同的层
    all_layers = set(normal_maps.keys()) | set(intervened_maps.keys())
    
    for layer_idx in sorted(all_layers):
        if layer_idx in normal_maps and layer_idx in intervened_maps:
            normal_map = normal_maps[layer_idx]
            intervened_map = intervened_maps[layer_idx]
            
            # 检查尺寸是否匹配，如果不匹配则跳过比较
            if normal_map.shape != intervened_map.shape:
                print(f"第 {layer_idx} 层的注意力图尺寸不匹配: 正常推理 {normal_map.shape} vs 干预推理 {intervened_map.shape}，跳过比较")
                continue
            
            # 计算差异图
            diff_map = intervened_map - normal_map
            
            # 为每层创建单独的子目录
            layer_save_path = os.path.join(save_path, f"layer_{layer_idx}")
            if not os.path.exists(layer_save_path):
                os.makedirs(layer_save_path)
            
            # 可视化正常注意力图
            normal_vis = visualize_heatmap(normal_map, image)
            normal_save_path = os.path.join(layer_save_path, "normal.jpg")
            Image.fromarray(normal_vis).save(normal_save_path)
            
            # 可视化干预后注意力图
            intervened_vis = visualize_heatmap(intervened_map, image)
            intervened_save_path = os.path.join(layer_save_path, "intervened.jpg")
            Image.fromarray(intervened_vis).save(intervened_save_path)
            
            # 可视化差异图
            diff_vis = visualize_heatmap(diff_map, image)
            diff_save_path = os.path.join(layer_save_path, "difference.jpg")
            Image.fromarray(diff_vis).save(diff_save_path)
            
            # 计算统计信息
            normal_mean = np.mean(normal_map)
            intervened_mean = np.mean(intervened_map)
            diff_mean = np.mean(diff_map)
            
            # print(f"第 {layer_idx} 层:")
            # print(f"  正常注意力图平均值: {normal_mean:.6f}")
            # print(f"  干预注意力图平均值: {intervened_mean:.6f}")
            # print(f"  差异图平均值: {diff_mean:.6f}")
            print(f"  已保存比较图到: {layer_save_path}")

            # 计算1D差异
            diff_map_1d = (intervened_maps[layer_idx] - normal_maps[layer_idx]).flatten()
            
            # 为每一层创建一个新的图
            plt.figure() 
            plt.plot(diff_map_1d)
            plt.title(f"Layer {layer_idx} - 1D Difference")
            plt.xlabel("Image Feature Index")
            plt.ylabel("Attention Difference")
            plt.grid(True) # 添加网格线
            plt.ylim(-0.02, 0.045) # 可以统一y轴范围，便于比较
            
            # 保存单层图
            plt.savefig(os.path.join(layer_save_path, "difference_1d.png"))
            plt.close() # 关闭图，避免在内存中累积
            
        else:
            print(f"第 {layer_idx} 层在一种或两种情况下缺失")

def visualize_combined_heatmaps(normal_maps, intervened_maps, image, save_path):
    """可视化所有层的注意力图组合"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # # 组合正常注意力图
    # if normal_maps:
    #     # 检查所有图的尺寸是否一致
    #     normal_shapes = [normal_maps[k].shape for k in normal_maps.keys()]
    #     if len(set(normal_shapes)) > 1:
    #         print("警告：正常推理的注意力图尺寸不一致，无法组合")
    #     else:
    #         normal_avg = np.mean(list(normal_maps.values()), axis=0)
    #         normal_combined_vis = visualize_heatmap(normal_avg, image)
    #         normal_combined_save_path = os.path.join(save_path, "combined_normal.jpg")
    #         Image.fromarray(normal_combined_vis).save(normal_combined_save_path)
    #         print(f"已保存组合正常注意力图到: {normal_combined_save_path}")
    
    # # 组合干预后注意力图
    # if intervened_maps:
    #     # 检查所有图的尺寸是否一致
    #     intervened_shapes = [intervened_maps[k].shape for k in intervened_maps.keys()]
    #     if len(set(intervened_shapes)) > 1:
    #         print("警告：干预推理的注意力图尺寸不一致，无法组合")
    #     else:
    #         intervened_avg = np.mean(list(intervened_maps.values()), axis=0)
    #         intervened_combined_vis = visualize_heatmap(intervened_avg, image)
    #         intervened_combined_save_path = os.path.join(save_path, "combined_intervened.jpg")
    #         Image.fromarray(intervened_combined_vis).save(intervened_combined_save_path)
    #         print(f"已保存组合干预注意力图到: {intervened_combined_save_path}")
        
    # 组合差异图
    if normal_maps and intervened_maps:
        # 确保两个字典有相同的层且尺寸匹配
        common_layers = set(normal_maps.keys()) & set(intervened_maps.keys())
        matched_maps = []
        
        for layer_idx in common_layers:
            normal_map = normal_maps[layer_idx]
            intervened_map = intervened_maps[layer_idx]
            
            # 只有尺寸匹配的层才用于计算差异
            if normal_map.shape == intervened_map.shape:
                matched_maps.append((normal_map, intervened_map, layer_idx))
            else:
                print(f"第 {layer_idx} 层尺寸不匹配: 正常推理 {normal_map.shape} vs 干预推理 {intervened_map.shape}，跳过差异计算")
        
        if matched_maps:
            diff_maps = [intervened_map - normal_map for normal_map, intervened_map, _ in matched_maps]
            diff_combined = np.mean(diff_maps, axis=0)
            diff_combined_vis = visualize_heatmap(diff_combined, image)
            diff_combined_save_path = os.path.join(save_path, "combined_difference.jpg")
            Image.fromarray(diff_combined_vis).save(diff_combined_save_path)
            print(f"已保存组合差异注意力图到: {diff_combined_save_path}")
        else:
            print("没有尺寸匹配的注意力图对，无法生成组合差异图")

def get_image_grid_size(original_image_size, vision_config):
    """
    根据模型配置，手动计算出经过保持长宽比缩放后的有效网格尺寸。
    """
    orig_w, orig_h = original_image_size
    # ======================= [核心修改点] =======================
    # 根据对processor行为的最终确认，目标尺寸是672
    # 根据打印出的config，patch_size是14
    target_size = 672
    try:
        patch_size = vision_config.patch_size
    except AttributeError:
        print("致命错误: 无法从vision_config中找到 'patch_size'。")
        return None
    # =========================================================

    if orig_w > orig_h:
        long_edge, short_edge = orig_w, orig_h
    else:
        long_edge, short_edge = orig_h, orig_w

    scale = target_size / long_edge
    scaled_short_edge = int(scale * short_edge)

    if orig_w > orig_h:
        scaled_w, scaled_h = target_size, scaled_short_edge
    else:
        scaled_w, scaled_h = scaled_short_edge, target_size

    grid_h = scaled_h // patch_size
    grid_w = scaled_w // patch_size
    
    return (grid_h, grid_w)

def visualize_bbox_on_image(image, bbox, save_path):
    """在图像上可视化bbox位置"""
    # 复制图像以避免修改原始图像
    image_array = np.array(image)
    image_copy = image_array.copy()
    
    # 获取图像尺寸
    h, w = image_copy.shape[:2]
    
    # 处理单个或多个bbox
    bboxes = bbox if isinstance(bbox, list) and (len(bbox) == 0 or isinstance(bbox[0], list)) else [bbox]
    
    # 为每个bbox绘制矩形
    for i, single_bbox in enumerate(bboxes):
        # bbox格式: [x1, y1, x2, y2]，值在0-1之间
        x1, y1, x2, y2 = single_bbox
        
        # 将相对坐标转换为像素坐标
        x1_pixel = int(x1 * w)
        y1_pixel = int(y1 * h)
        x2_pixel = int(x2 * w)
        y2_pixel = int(y2 * h)
        
        # 在图像上绘制bbox（使用不同颜色区分多个框）
        color = (0, 255, 0)  # 默认绿色
        if i > 0:
            # 为不同的框使用不同的颜色
            colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[(i - 1) % len(colors)]
        
        cv2.rectangle(image_copy, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), color, 2)
        
        # 添加框的序号
        cv2.putText(image_copy, f'{i+1}', (x1_pixel, y1_pixel-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # 保存图像
    bbox_image_path = os.path.join(save_path, "bbox_location.jpg")
    Image.fromarray(image_copy).save(bbox_image_path)
    print(f"已保存bbox位置可视化图像到: {bbox_image_path}")


def resize_image_by_pixel_limit(image: Image.Image, max_pixels: int = 1400000) -> Image.Image:
    """
    检查图像的总像素数，如果超过限制，则按比例缩小图像。

    Args:
        image (Image.Image): 输入的PIL图像对象。
        max_pixels (int): 允许的最大总像素数。
                          [注意] 此阈值已针对“注意力干预”模式进行调整。
                          干预模式比正常推理需要更多显存，因此使用更保守的 180 万像素
                          （约 1340x1340）作为安全阈值。
                          原正常推理阈值为 210 万。

    Returns:
        Image.Image: 经过可能缩放后的图像对象。
    """
    width, height = image.size
    current_pixels = width * height

    if current_pixels > max_pixels:
        # 计算缩放比例
        scale_factor = math.sqrt(max_pixels / current_pixels)
        
        # 计算新的尺寸
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"  - [注意] 图片尺寸 {width}x{height} (总像素: {current_pixels:,}) 超出干预安全阈值。")
        print(f"  - 按比例缩小至 -> {new_width}x{new_height} (总像素: {new_width*new_height:,})")
        
        # 使用LANCZOS高质量抗锯齿缩放
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 如果未超过阈值，返回原图
    return image