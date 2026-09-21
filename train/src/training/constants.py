IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

ANCHOR_START_TOKEN = "<|anchor_start|>"
ANCHOR_END_TOKEN = "<|anchor_end|>"

DET_PAD_TOKEN = "<|det_pad|>"
LAYOUT_PAD_TOKEN = "<|layout_pad|>"
FLOW_PAD_TOKEN = "<|flow_pad|>"

# 索引槽位 token：相同 token 重复 k 次时模型须隐式计数已发个数（Transformer 弱项），
# 改为每槽独立 token 后计数变成显式状态，停止决策退化为"还要不要更多"的二分类；
# 且各槽位有独立 embedding，可分化语义（det_1 出现在全部样本，天然由粗到细）。
MAX_ANCHOR_SLOTS = 8
DET_SLOT_TOKENS = [f"<|det_{i}|>" for i in range(1, MAX_ANCHOR_SLOTS + 1)]
LAYOUT_SLOT_TOKENS = [f"<|layout_{i}|>" for i in range(1, MAX_ANCHOR_SLOTS + 1)]
FLOW_SLOT_TOKENS = [f"<|flow_{i}|>" for i in range(1, MAX_ANCHOR_SLOTS + 1)]
ANCHOR_SLOT_TOKENS = {
    "det": DET_SLOT_TOKENS,
    "layout": LAYOUT_SLOT_TOKENS,
    "flow": FLOW_SLOT_TOKENS,
}

SYSTEM_MESSAGE = "You are a helpful assistant."