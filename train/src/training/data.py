import copy
import os
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import AutoImageProcessor
import re
import numpy as np
import cv2
from torchvision import transforms
import random

# 注意：这里引用了上一层目录的 constants，请确保路径正确
from .params import DataArguments
from .constants import *

# === 新增：Doc-CoVT 专用模板与指令 ===

# 注：CoT 由 build_doc_cot 按自适应预算动态拼接（见下方 DOC_COT_STEPS）。
# 不再使用带 Step 1/2/3 编号的固定模板——det/layout/flow 是并列的感知通道，
# 编号暗示时序递进，整步省略时会出现"Step 1 跳到 Step 3"的自相矛盾。

# === 自适应视觉 token 预算 ===
# token 数应正比于该分支要表达的信息量；信息平凡时整步省略（分支级选择）。
# 预算由 tool/build_dataset_cache.py 从 teacher 信号离线算出（逻辑在 build_anchor_budget.py），此处按图查表。
# 未配置 ANCHOR_BUDGET_FILE 时回落固定 4/4/4，与原方案完全一致。
DEFAULT_ANCHOR_K = 4
_ANCHOR_BUDGET = None

DOC_COT_STEPS = (
    ("det", "Text regions:"),
    ("layout", "Layout structure:"),
    ("flow", "Reading flow:"),
)
_STEP_PAD_TOKEN = {"det": DET_PAD_TOKEN, "layout": LAYOUT_PAD_TOKEN, "flow": FLOW_PAD_TOKEN}


def _get_anchor_budget():
    global _ANCHOR_BUDGET
    if _ANCHOR_BUDGET is None:
        path = os.environ.get("ANCHOR_BUDGET_FILE")
        if not path:
            _ANCHOR_BUDGET = {}
        else:
            with open(path) as f:
                _ANCHOR_BUDGET = json.load(f)
            print(f"[Data] 自适应锚点预算已加载: {len(_ANCHOR_BUDGET)} 张图")
    return _ANCHOR_BUDGET


def get_anchor_budget(image_file):
    """返回该图各分支的 token 数；无预算表或未命中时回落固定 4/4/4。"""
    table = _get_anchor_budget()
    if table and isinstance(image_file, str):
        b = table.get(os.path.basename(image_file))
        if b:
            return {k: int(b.get(k, 0)) for k, _ in DOC_COT_STEPS}
    return {k: DEFAULT_ANCHOR_K for k, _ in DOC_COT_STEPS}


def _use_indexed_tokens():
    return os.environ.get("ANCHOR_INDEXED_TOKENS", "0") == "1"


def _anchor_token_str(name, k):
    """该分支发出的 token 串：索引模式下为 slot_1..slot_k，否则为同一 pad 重复 k 次。"""
    if _use_indexed_tokens():
        slots = ANCHOR_SLOT_TOKENS[name]
        return "".join(slots[:min(k, len(slots))])
    return _STEP_PAD_TOKEN[name] * k


def build_doc_cot(budget):
    """按预算拼接 <think> 块：k=0 的步骤整步省略。"""
    lines = [
        f"{text} {_anchor_token_str(name, budget[name])}"
        for name, text in DOC_COT_STEPS if budget.get(name, 0) > 0
    ]
    return "<think>\n" + "".join(l + "\n" for l in lines) + "</think>"

# Stage 1 专用的纯视觉指令池
STAGE1_INSTRUCTIONS = [
    "Analyze the visual structure of this document.",
    "Perform a comprehensive visual analysis.",
    "Detect text, layout, and reading order.",
    "Generate the visual perception maps for this image.",
    "Show me the visual understanding of this document.",
    "Visualize the document structure."
]

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps):
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]
    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    return video_input[0], video_kwargs

# === Doc-CoVT 核心数据构造函数 ===

def get_stage1_data(user_input_content, user_role, gpt_role, image_file=None):
    """
    Stage 1 (Visual Alignment):
    忽略 User 原始问题，替换为视觉指令。
    Assistant 只输出思维链 (Visual Tokens)，没有文本 Answer。
    """
    # 1. 准备指令
    instruction = random.choice(STAGE1_INSTRUCTIONS)
    
    # 2. 构造 Assistant 回复 (纯思维链)，token 数与出现的步骤由自适应预算决定
    cot_response = build_doc_cot(get_anchor_budget(image_file))
    
    # 3. 构造 User 输入 (保留 Image Token，替换 Text)
    # user_input_content 格式通常是: <|vision_start|>...<|vision_end|>Original Question
    if VISION_END_TOKEN in user_input_content:
        # 分割出图片部分
        image_part = user_input_content.split(VISION_END_TOKEN)[0] + VISION_END_TOKEN
        new_user_content = image_part + "\n" + instruction
    else:
        # 如果没有图片token (异常情况)，直接用指令
        new_user_content = instruction

    # 4. 组装 ChatML 格式
    new_user_text = f"{DEFAULT_IM_START_TOKEN}{user_role}\n{new_user_content}\n{DEFAULT_IM_END_TOKEN}\n"
    new_gpt_text = f"{DEFAULT_IM_START_TOKEN}{gpt_role}\n{cot_response}\n{DEFAULT_IM_END_TOKEN}\n"
    
    return new_user_text, new_gpt_text

def get_stage2_data(gpt_response_content, image_file=None):
    """
    Stage 2/3 (Joint Reasoning):
    Assistant 输出 = 思维链 + 原始 Answer。
    """
    # 1. 准备思维链，token 数与出现的步骤由自适应预算决定
    cot_block = build_doc_cot(get_anchor_budget(image_file))
    
    # 2. 拼接 Answer
    final_response = f"{cot_block}\n<answer> {gpt_response_content} </answer>"
    
    return final_response

def get_token_num(anchor_model_id):
    token_nums = []
    for anchor_model in anchor_model_id:
        if anchor_model in ["det", "layout", "flow"]:
            token_nums.append(4)
    return token_nums

def get_anchor_token(anchor_model_id):
    anchor_tokens = []
    for anchor_model in anchor_model_id:
        if anchor_model == "det": anchor_tokens.append(DET_PAD_TOKEN)
        elif anchor_model == "layout": anchor_tokens.append(LAYOUT_PAD_TOKEN)
        elif anchor_model == "flow": anchor_tokens.append(FLOW_PAD_TOKEN)
    return anchor_tokens

def get_anchor_task_name(anchor_model_id):
    anchor_task_names = []
    for anchor_model in anchor_model_id:
        if anchor_model == "det": anchor_task_names.append("text detection map")
        elif anchor_model == "layout": anchor_task_names.append("document layout map")
        elif anchor_model == "flow": anchor_task_names.append("reading flow map")
        # ... others ...
    return anchor_task_names

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        shuffle=True,
        random_seed=42,
        anchor_model_id=None,
        stage_1_steps=0,
        per_device_batch_size=1,
        grad_accum_steps=1,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps
        self.anchor_model_id = anchor_model_id
        
        # 辅助信息 (虽然在这里可能用不到了，但保留以防万一)
        self.anchor_token_nums = get_token_num(anchor_model_id)
        self.anchor_tokens = get_anchor_token(anchor_model_id)
        self.anchor_task_names = get_anchor_task_name(anchor_model_id)
        
        # === 步数同步机制（确定性位置计算，与 prefetch 无关）===
        # 样本属于哪个 optimizer step 由它在数据流中的消费位置唯一确定：
        #   DataLoader 按 round-robin 把 batch 分给 worker，故
        #   global_batch = local_batch * num_workers + worker_id
        #   opt_step = epoch_start_step + global_batch // grad_accum_steps
        # 而不是按“生成时刻”的共享步数决策（后者会被 worker 预取提前量冲掉阶段边界）。
        # epoch_start_step 由 StepSyncCallback 在每个 epoch 开始时写入（resume 安全）；
        # worker 每个 epoch 重新 fork，_local_samples 自动归零。
        self._epoch_start_step = mp.Value('q', 0)
        self._local_samples = 0
        self._last_is_stage1 = None
        self.per_device_batch_size = per_device_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.stage_1_steps = stage_1_steps
        
        self.rng = np.random.default_rng(seed=random_seed)
        if shuffle:
            self.rng.shuffle(self.list_data_dict)
    
    def set_epoch_start_step(self, step):
        with self._epoch_start_step.get_lock():
            self._epoch_start_step.value = step
        # num_workers=0 时主进程直接取数，需在 epoch 边界重置位置计数
        self._local_samples = 0

    @property
    def cur_step(self):
        info = torch.utils.data.get_worker_info()
        local_batch = self._local_samples // self.per_device_batch_size
        if info is None:
            global_batch = local_batch
        else:
            global_batch = local_batch * info.num_workers + info.id
        return self._epoch_start_step.value + global_batch // self.grad_accum_steps
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        sources = self.list_data_dict[i]
        is_video = False
        processor = self.processor
        
        # === 1. 处理图像/视频加载 ===
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files] # 统一转list
                
            # 加载并预处理图片
            loaded_images = []
            final_image_files = [] # 存储PIL对象或路径
            
            for image_file in image_files:
                if isinstance(image_file, str):
                    if not os.path.exists(image_file) and image_folder:
                        image_file = os.path.join(image_folder, image_file)
                    img_obj = Image.open(image_file).convert("RGB")
                    # 传路径而非 PIL：作为 teacher 磁盘缓存的键，且避免 worker→主进程搬运像素
                    final_image_files.append(image_file)
                else:
                    img_obj = image_file.convert("RGB")
                    final_image_files.append(img_obj)
                
                loaded_images.append(get_image_info(img_obj, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
            
            images = loaded_images
            
        elif "video" in sources:
            # (省略视频处理逻辑，保持原样)
            is_video = True
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            # ...
            videos = [] # Placeholder
            final_image_files = [] 
        else:
            # 纯文本数据 (通常不应该进入这里，除非混合训练)
            grid_key = None
            pixel_key = None
            images = None
            videos = None
            final_image_files = []

        if images is None and not is_video:
            # Fallback
            black_image = Image.new("RGB", (self.image_resized_w or 256, self.image_resized_h or 256), (0, 0, 0))
            images = [get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h)]
            final_image_files = [black_image]

        # === 2. 处理对话文本 ===
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # System Message
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            sys_tokens = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            sys_labels = torch.full_like(sys_tokens, IGNORE_INDEX) 
            all_input_ids.append(sys_tokens.squeeze(0))
            all_labels.append(sys_labels.squeeze(0))
            
        # 遍历对话轮次 (通常 DocVQA 是单轮)
        for _, j in enumerate(range(0, len(sources), 2)):
            if j >= 2: break # 限制为单轮
            
            user_input = sources[j]
            gpt_response = sources[j + 1]
            
            # === 核心逻辑修改：Doc-CoVT 阶段控制 ===
            
            # 只有当包含图片时，才触发视觉思维链
            if DEFAULT_IMAGE_TOKEN in user_input['content']:
                
                cur_step = self.cur_step
                is_stage1 = cur_step < self.stage_1_steps
                
                cur_img = final_image_files[0] if final_image_files else None

                # 分支 1: Stage 1 - Visual Alignment (纯视觉训练)
                if is_stage1:
                    user_text, gpt_text = get_stage1_data(
                        user_input['content'], user_input['role'], gpt_response['role'],
                        image_file=cur_img
                    )
                    
                # 分支 2: Stage 2 - Joint Reasoning (CoT + QA)
                # 分支 3: Stage 3 - Robustness (保持 CoT，通过 Loss 调节)
                else: 
                    # 获取处理后的回答内容 (CoT + Answer)
                    new_gpt_content = get_stage2_data(gpt_response['content'], image_file=cur_img)
                    
                    # 构造标准 ChatML 格式
                    user_text = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                    gpt_text = f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{new_gpt_content}\n{DEFAULT_IM_END_TOKEN}\n"

                # 数据阶段切换时打印一次（每个 worker 各自打印），便于核对切换点
                if self._last_is_stage1 is not None and is_stage1 != self._last_is_stage1:
                    print(f"\n[Data] Stage {'2->1' if is_stage1 else '1->2'} switch at step {cur_step}, sample:\n{gpt_text}\n")
                self._last_is_stage1 = is_stage1

                # 打印日志 (每100步打印一次，避免 I/O 阻塞训练)
                if cur_step % 100 == 0:
                    print(f"\n[Step {cur_step}] Data Sample:")
                    print(f"User: \n{user_text}")
                    print(f"GPT:  \n{gpt_text}")
                    print("-" * 30)

                # Tokenize (text 已经包含 special tokens，所以 add_special=False)
                inputs = processor(text=[user_text], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                
                # 收集视觉特征
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                if "second_per_grid_ts" in inputs:
                    all_second_gird.extend(inputs["second_per_grid_ts"])

                # Tokenize Response
                response_input_ids = processor.tokenizer(gpt_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            # 纯文本数据 (Fallback)
            else:
                user_text = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                gpt_text = f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                prompt_input_ids = processor.tokenizer(user_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
                response_input_ids = processor.tokenizer(gpt_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            # 拼接 input_ids 和 labels
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # 拼接所有轮次
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long) # 全 1 mask
        
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            # 传递 PIL 对象给 collate_fn 或 model，用于 Teacher Inference
            data_dict["image_files"] = final_image_files 

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird
        
        # 位置计数：本 worker 已产出的样本数
        self._local_samples += 1
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        batch_image_files = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            if "image_files" in keys:
                batch_image_files.append(example["image_files"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
            
        if len(batch_image_files) > 0:
            data_dict["image_files"] = batch_image_files

        return data_dict

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args, anchor_model_id, stage_1_steps=0, per_device_batch_size=1, grad_accum_steps=1):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id, anchor_model_id=anchor_model_id,
        stage_1_steps=stage_1_steps, per_device_batch_size=per_device_batch_size, grad_accum_steps=grad_accum_steps
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)