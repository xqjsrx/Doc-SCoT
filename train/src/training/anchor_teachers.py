"""Doc-CoVT teacher 侧：det/layout/flow 三分支的教师模型与对齐 loss。

- det:    doctr DBNet(ResNet50)，hook FPN neck 特征（解码场）+ head 概率图（GT）
- layout: DocLayout-YOLO(DocStructBench)，neck 32x32 特征 + 语义灰度光栅 GT
- flow:   DBNet neck 特征 + 8 坐标通道（零额外推理）；阅读顺序主路径为
          词级 LayoutReader 聚合到块级，回退链：块级 LayoutReader → 启发式几何排序

teacher_cache 磁盘缓存存"原料"（特征/框/词序）而非成品 GT，
命中则跳过全部在线 teacher 推理，未命中自动回退在线计算。
"""
import huggingface_hub

# doctr 1.0.0 顶层 import 了 huggingface_hub>=1.x 已移除的符号（仅 push-to-hub 功能用到，
# 训练/推理路径不触及）；必须在 import doctr 之前打补丁
for _name in ("Repository", "get_token_permission"):
    if not hasattr(huggingface_hub, _name):
        setattr(huggingface_hub, _name, None)

import os
import math
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from doctr.models import db_resnet50
from doclayout_yolo import YOLOv10

# === LayoutReader (阅读顺序预测模型) ===
from transformers import LayoutLMv3ForTokenClassification

# 教师模型位置：默认指向本地目录，可用环境变量覆盖（也可填 HuggingFace repo id）
LAYOUTREADER_MODEL_PATH = os.environ.get(
    "LAYOUTREADER_MODEL_PATH", "hfl/layoutreader")
LAYOUTREADER_MAX_LEN = 510
LAYOUTREADER_CLS_TOKEN_ID = 0
LAYOUTREADER_UNK_TOKEN_ID = 3
LAYOUTREADER_EOS_TOKEN_ID = 2

LAYOUT_MODEL_PATH = os.environ.get(
    "LAYOUT_MODEL_PATH", "juliozhao/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt")


def _layoutreader_boxes2inputs(boxes):
    """将 bbox 列表转换为 LayoutReader 模型输入格式"""
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [LAYOUTREADER_CLS_TOKEN_ID] + [LAYOUTREADER_UNK_TOKEN_ID] * len(boxes) + [LAYOUTREADER_EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def _layoutreader_prepare_inputs(inputs, model):
    """将输入张量移动到模型设备和数据类型"""
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret


def _layoutreader_parse_logits(logits, length):
    """
    解析模型 logits 为阅读顺序排列
    :param logits: 模型输出的 logits [seq_len, vocab_len]
    :param length: 输入框数量
    :return: orders 列表, orders[i] 表示第 i 个框在阅读序列中的位置 (0-based)
    """
    from collections import defaultdict
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()
    return ret


def layoutreader_predict_reading_order(model, boxes_norm01, image_width, image_height):
    """
    使用 LayoutReader 模型预测阅读顺序
    :param model: LayoutLMv3ForTokenClassification 模型
    :param boxes_norm01: 归一化到 0-1 的 bbox 列表 [[x1,y1,x2,y2], ...]
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: sorted_indices 列表, 按阅读顺序排列的原始索引
    """
    if len(boxes_norm01) == 0:
        return []

    # 截断到 MAX_LEN
    if len(boxes_norm01) > LAYOUTREADER_MAX_LEN:
        boxes_norm01 = boxes_norm01[:LAYOUTREADER_MAX_LEN]

    # 归一化到 0-1000 范围 (LayoutReader 要求)
    # boxes_norm01 是 0-1 归一化的，所以乘以 1000 即可
    norm_boxes = []
    for x1, y1, x2, y2 in boxes_norm01:
        left = round(x1 * 1000)
        top = round(y1 * 1000)
        right = round(x2 * 1000)
        bottom = round(y2 * 1000)
        right = max(right, left)
        bottom = max(bottom, top)
        right = min(right, 1000)
        bottom = min(bottom, 1000)
        left = min(left, 1000)
        top = min(top, 1000)
        norm_boxes.append([left, top, right, bottom])

    # 模型推理
    inputs = _layoutreader_boxes2inputs(norm_boxes)
    inputs = _layoutreader_prepare_inputs(inputs, model)
    with torch.no_grad():
        logits = model(**inputs).logits.cpu().squeeze(0)
    orders = _layoutreader_parse_logits(logits, len(norm_boxes))

    # orders[i] 表示第 i 个框在阅读序列中的位置
    # sorted_indices 按阅读顺序排列
    sorted_indices = sorted(range(len(orders)), key=lambda i: orders[i])
    return sorted_indices


def get_reading_order_indices(boxes, y_threshold=10):
    """
    基于几何位置的阅读顺序排序（启发式回退）。
    Args:
        boxes: List of [x1, y1, x2, y2] (绝对坐标或相对坐标均可)
        y_threshold: 判定同一行的垂直容差
    Returns:
        sorted_indices: 排序后的索引列表 [idx_1, idx_2, ...]
    """
    if len(boxes) == 0:
        return []

    # 构造带索引的对象: {'box': [x1,y1,x2,y2], 'index': i, 'center_y': y_mid}
    box_objs = []
    for i, b in enumerate(boxes):
        box_objs.append({
            'box': b,
            'index': i,
            'y_center': (b[1] + b[3]) / 2,
            'y_top': b[1]
        })

    # 1. 全局先按 Top Y 排序
    box_objs.sort(key=lambda x: x['y_top'])

    lines = []
    current_line = []

    # 2. 聚类成行
    for obj in box_objs:
        if not current_line:
            current_line.append(obj)
            continue

        # 判断是否属于当前行:
        # 如果当前框的中心Y 与 当前行平均中心Y 的差距小于阈值，视为同一行
        line_avg_y = sum(o['y_center'] for o in current_line) / len(current_line)
        box_height = obj['box'][3] - obj['box'][1]

        # 这里的阈值可以是 box高度的一半
        if abs(obj['y_center'] - line_avg_y) < (box_height * 0.5):
            current_line.append(obj)
        else:
            lines.append(current_line)
            current_line = [obj]

    if current_line:
        lines.append(current_line)

    # 3. 行内按 Left X 排序，并收集索引
    final_indices = []
    for line in lines:
        # 按 x1 排序
        line.sort(key=lambda x: x['box'][0])
        final_indices.extend([x['index'] for x in line])

    return final_indices


class AnchorLoss():
    def __init__(self, anchor_loss_weight=None):
        # anchor_loss_weight 仅为签名兼容保留，权重硬编码 1.0
        self.det_loss_weight = 1.0
        self.layout_loss_weight = 1.0
        self.flow_loss_weight = 1.0
        self.loss_fn = nn.MSELoss()

    def get_det_loss(self, pred_map, gt_map):
        if self.det_loss_weight > 0:
            # 1. 尺寸对齐
            if pred_map.shape != gt_map.shape:
                gt_map = F.interpolate(gt_map, size=pred_map.shape[2:], mode='bilinear', align_corners=False)

            # 2. 数值截断：范围必须是 [0, 1]，因为 Sigmoid 输出是 [0, 1]
            gt_map = torch.clamp(gt_map, 0.0, 1.0)
            pred_map = torch.clamp(pred_map, 0.0, 1.0)

            # 3. MSE：误差小时梯度迅速趋零，数值稳定；x200 放大到与 text loss 可比量级
            det_loss = self.loss_fn(pred_map, gt_map)
            return det_loss * self.det_loss_weight * 200.0
        else:
            return 0.

    def get_layout_loss(self, pred_map, gt_map):
        if self.layout_loss_weight > 0:
            if pred_map.shape != gt_map.shape:
                gt_map = F.interpolate(gt_map, size=pred_map.shape[2:], mode='bilinear', align_corners=False)

            # GT 是 0.3~0.9 语义灰度（非 0/1），MSE + L1 组合
            mse_loss = self.loss_fn(pred_map, gt_map)
            l1_loss = F.l1_loss(pred_map, gt_map)
            loss = 0.5 * mse_loss + 0.5 * l1_loss

            return loss * 50.0
        else:
            return 0.

    def get_flow_loss(self, pred_map, sil_gt, boxes_in_order=None):
        """
        框级 pairwise ranking + 轻量剪影 L1
        - ranking 项：监督"阅读序靠后的框在 pred_map 上更亮"的相对关系，
          与框数 N 无关、背景不参与、不要求绝对亮度
        - 剪影项：框内=1/背景=0 的 L1，保留定位能力（小权重）
        - 框数 <2 时没有顺序信息，ranking 项自动为 0（保持计算图连接），只学剪影
        :param pred_map: [1,1,H,W] sigmoid 后的预测图
        :param sil_gt:   [1,1,128,128] 剪影 GT（框内 1）
        :param boxes_in_order: 按阅读顺序排列的 xyxyn 框列表
        """
        if self.flow_loss_weight <= 0:
            return 0.

        if pred_map.shape != sil_gt.shape:
            sil_gt = F.interpolate(sil_gt, size=pred_map.shape[2:], mode='bilinear', align_corners=False)
        sil_loss = F.l1_loss(torch.clamp(pred_map, 0.0, 1.0), torch.clamp(sil_gt, 0.0, 1.0))

        # 保持与计算图连接的零（N<2 时 ranking 项不参与）
        rank_loss = pred_map.sum() * 0.0
        n = len(boxes_in_order) if boxes_in_order else 0
        if n >= 2:
            H, W = pred_map.shape[-2:]
            pm = pred_map[0, 0]
            means = []
            for x1, y1, x2, y2 in boxes_in_order:
                xs = min(max(int(x1 * W), 0), W - 1)
                ys = min(max(int(y1 * H), 0), H - 1)
                xe = min(max(int(x2 * W), xs + 1), W)
                ye = min(max(int(y2 * H), ys + 1), H)
                means.append(pm[ys:ye, xs:xe].mean())
            means = torch.stack(means)

            # 确定性配对：相邻对（排序的充分约束）+ 跨 2 对（增强传递性）
            pairs = [(i, i + 1) for i in range(n - 1)]
            if n > 2:
                pairs += [(i, i + 2) for i in range(n - 2)][:32]

            # 自适应 margin：sigmoid 值域内要容纳 N 个框的排序
            margin = min(0.05, 0.5 / (n - 1))
            viol = torch.stack([F.relu(margin - (means[j] - means[i])) for i, j in pairs])
            rank_loss = viol.mean()

        return (rank_loss + 0.3 * sil_loss) * self.flow_loss_weight * 50.0


class AnchorModels():
    def __init__(self, anchor_model_id):
        self.anchor_model_id = anchor_model_id
        self.teacher_cache_dir = None

        # 存储 Hook 捕获的数据
        self.det_storage = {
            'neck': None,  # FPN 输出的特征
            'head': None   # Head 输出的 GT
        }
        self.dbnet = None

        # === 按图推理缓存 ===
        # 同一张图在一次 forward 内会被 embed 提取与 GT 生成多次访问，
        # 缓存让每个 teacher 对每张图只推理一次（weakref 校验对象身份，
        # 防止旧 PIL 对象被 GC 后 id 复用导致脏命中）
        self._det_cache = {}
        self._layout_cache = {}

        # === 1. Text Detection Expert (Doctr DBNet) ===
        if "det" in self.anchor_model_id:
            print("[Doc-CoVT] Initializing Teacher: DBNet (ResNet50) from Doctr...")
            try:
                self.dbnet = db_resnet50(pretrained=True, assume_straight_pages=False).eval()
            except Exception as e:
                print(f"[Error] Failed to load db_resnet50: {e}")
                raise e

            # Doctr DBNet 的标准结构: feat_extractor -> fpn -> prob_head
            if hasattr(self.dbnet, 'feat_extractor'):
                self.det_backbone = self.dbnet.feat_extractor
            else:
                raise AttributeError("DBNet missing 'feat_extractor'")

            if hasattr(self.dbnet, 'fpn'):
                self.det_neck = self.dbnet.fpn
            else:
                raise AttributeError("DBNet missing 'fpn'")

            if hasattr(self.dbnet, 'prob_head'):
                self.det_head = self.dbnet.prob_head
            else:
                print("DBNet missing 'prob_head', available children:")
                for n, _ in self.dbnet.named_children(): print(n)
                raise AttributeError("DBNet missing 'prob_head'")

            # Hook 1: 截获 FPN 输出 (作为 VLM 的视觉特征)
            def get_neck_hook(module, input, output):
                # FPN 输出通常直接就是 Tensor [B, 256, H/4, W/4]
                self.det_storage['neck'] = output

            # Hook 2: 截获 Head 输出 (作为 GT)
            def get_head_hook(module, input, output):
                # prob_head 输出通常直接是 Tensor [B, 1, H, W]
                self.det_storage['head'] = output

            # Hook FPN 的输出，而不是 Backbone 的输出，因为 FPN 特征更强
            self.det_neck.register_forward_hook(get_neck_hook)
            self.det_head.register_forward_hook(get_head_hook)

            # 预处理
            self.det_normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )

        # === 2. Layout Analysis Expert (DocLayout-YOLO) ===
        if "layout" in self.anchor_model_id or "flow" in self.anchor_model_id:
            print("[Doc-CoVT] Initializing Teacher: DocLayout-YOLO (DocStructBench)...")

            try:
                self.layout_model = YOLOv10(LAYOUT_MODEL_PATH)
            except Exception as e:
                print(f"[Error] Failed to load DocLayout-YOLO from {LAYOUT_MODEL_PATH}: {e}")
                self.layout_model = None

            if self.layout_model:
                self.layout_features = {}
                def get_layout_hook(module, input, output):
                    self.layout_features['neck'] = output

                # 访问底层 PyTorch module: model (Wrapper) -> model (PyTorch) -> model (Sequential) -> [6]
                try:
                    self.layout_model.model.model[6].register_forward_hook(get_layout_hook)
                except AttributeError:
                    print("[Error] Failed to hook YOLO layer 6. Check model architecture.")

                # DocLayout-YOLO DocStructBench 类别定义:
                # 0: title, 1: plain_text, 2: abandon, 3: figure, 4: figure_caption,
                # 5: table, 6: table_caption, 7: table_footnote, 8: isolate_formula, 9: formula_caption
                # 语义灰度映射：0.9 而非 1.0——预测经 sigmoid，1.0 不可达（恒定残差+梯度消失）
                self.layout_class_vals = {
                    0: 0.9,  # Title (标题 - 导航灯塔)
                    1: 0.5,  # Plain Text (正文 - 答案矿藏)
                    2: 0.0,  # Abandon (丢弃)
                    3: 0.3,  # Figure (图片 - 视觉元素)
                    4: 0.6,  # Figure Caption
                    5: 0.8,  # Table (表格 - 重点结构)
                    6: 0.6,  # Table Caption
                    7: 0.5,  # Table Footnote (同正文)
                    8: 0.8,  # Isolate Formula (公式 - 高密度)
                    9: 0.6,  # Formula Caption
                }
            else:
                self.layout_model = None

        # === 3. LayoutReader (阅读顺序预测模型) ===
        self.layoutreader = None
        if "flow" in self.anchor_model_id:
            print("[Doc-CoVT] Initializing Teacher: LayoutReader (LayoutLMv3)...")
            try:
                self.layoutreader = (
                    LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTREADER_MODEL_PATH)
                    .bfloat16()
                    .eval()
                )
                print(f"[Doc-CoVT] LayoutReader loaded from {LAYOUTREADER_MODEL_PATH}")
            except Exception as e:
                print(f"[Error] Failed to load LayoutReader: {e}")
                self.layoutreader = None

    def set_device(self, device):
        self.device = device
        if "det" in self.anchor_model_id:
            self.dbnet.to(device)
        if "layout" in self.anchor_model_id or "flow" in self.anchor_model_id:
            # YOLO Wrapper 这一步会自动把内部 model 转过去
            self.layout_model.to(device)
        if "flow" in self.anchor_model_id and self.layoutreader is not None:
            self.layoutreader.to(device)

    def set_float(self):
        if "det" in self.anchor_model_id:
            self.dbnet.float()
        if "layout" in self.anchor_model_id or "flow" in self.anchor_model_id:
            self.layout_model.float()
        if "flow" in self.anchor_model_id and self.layoutreader is not None:
            self.layoutreader.bfloat16()

    def _run_det_forward(self, image):
        """
        手动串联: Backbone -> (List) -> FPN -> Head
        """
        if self.dbnet is None: return

        # 1. 预处理
        target_size = (512, 512)
        img_resized = image.resize(target_size)
        x = transforms.ToTensor()(img_resized)
        x = self.det_normalize(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 2. Backbone 返回 OrderedDict，FPN 期待 List[Tensor]，取 values() 转 list
            feats_dict = self.det_backbone(x)
            feats_list = list(feats_dict.values())

            # 3. FPN (触发 get_neck_hook)
            fpn_out = self.det_neck(feats_list)

            # 4. Head (触发 get_head_hook)
            _ = self.det_head(fpn_out)

        # 此时 self.det_storage 已填充

    # ==================== Teacher 磁盘缓存与路径入参支持 ====================
    # image 参数同时支持 PIL 对象（评测/RL 脚本）与图片路径 str（训练）：
    # 路径入参时以 basename 为键命中 teacher_cache_dir 下的预计算文件，
    # 命中则完全跳过 DBNet/YOLO/LayoutReader 推理；未命中自动回退在线计算。

    class _CachedResult:
        """磁盘缓存命中时替代 ultralytics Results 的轻量 shim，
        只暴露运行时用到的 boxes.xyxyn / boxes.cls / len()"""
        class _B:
            def __init__(self, xyxyn, cls):
                self.xyxyn, self.cls = xyxyn, cls
            def __len__(self):
                return int(self.xyxyn.shape[0])
        def __init__(self, xyxyn, cls):
            self.boxes = self._B(xyxyn, cls) if xyxyn is not None and xyxyn.shape[0] > 0 else None

    def _img_key(self, image):
        return image if isinstance(image, str) else id(image)

    def _to_pil(self, image):
        """仅在需要真实像素（在线 teacher 推理）时才打开路径"""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        return image

    def _cache_get(self, cache, image):
        entry = cache.get(self._img_key(image))
        if entry is None:
            return None
        ref, outputs = entry
        # 路径键稳定无需校验；PIL 键需 weakref 验身份（防 id 复用）
        if ref is not None and ref() is not image:
            return None
        return outputs

    def _cache_put(self, cache, image, outputs, max_entries=8):
        # 先清理已被 GC 的 PIL 条目，仍超限则整体清空（缓存只需覆盖一次 forward 内的 batch）
        dead = [k for k, (ref, _) in cache.items() if ref is not None and ref() is None]
        for k in dead:
            cache.pop(k)
        if len(cache) >= max_entries:
            cache.clear()
        ref = None if isinstance(image, str) else weakref.ref(image)
        cache[self._img_key(image)] = (ref, outputs)

    def _load_teacher_cache(self, image):
        """命中磁盘缓存时一次性回填 det/layout 两个内存缓存，返回是否命中"""
        cache_dir = getattr(self, 'teacher_cache_dir', None)
        if not cache_dir or not isinstance(image, str):
            return False
        f = os.path.join(cache_dir, os.path.basename(image) + '.pt')
        if not os.path.exists(f):
            return False
        try:
            blob = torch.load(f, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f'[TeacherCache] 读取失败 {f}: {e}')
            return False
        det_outputs = {
            'neck': blob['det_neck'].unsqueeze(0).float(),
            'head': blob['det_head'].unsqueeze(0).float(),
        }
        self._cache_put(self._det_cache, image, det_outputs)
        boxes = blob['boxes'].to(self.device)
        cls = blob['cls'].to(self.device)
        layout_outputs = {
            'feat': blob['layout_feat'].unsqueeze(0).float(),
            'result': self._CachedResult(boxes, cls),
            'order': blob.get('order', None),
        }
        if isinstance(layout_outputs['order'], torch.Tensor):
            layout_outputs['order'] = [int(i) for i in layout_outputs['order']]
        self._cache_put(self._layout_cache, image, layout_outputs)
        return True

    def _get_det_outputs(self, image):
        """单张图只执行一次 DBNet 推理，neck/head 按图缓存供 embed 与 GT 复用；
        优先磁盘预计算缓存，未命中时在线计算"""
        outputs = self._cache_get(self._det_cache, image)
        if outputs is not None:
            return outputs
        if self._load_teacher_cache(image):
            return self._cache_get(self._det_cache, image)
        self._run_det_forward(self._to_pil(image))
        outputs = {
            'neck': self.det_storage['neck'].detach() if self.det_storage['neck'] is not None else None,
            'head': self.det_storage['head'].detach() if self.det_storage['head'] is not None else None,
        }
        self._cache_put(self._det_cache, image, outputs)
        return outputs

    def get_det_embed(self, image):
        if "det" in self.anchor_model_id:
            try:
                feat = self._get_det_outputs(image)['neck']
                if feat is not None:
                    return feat
            except Exception as e:
                print(f"[Error in get_det_embed] {e}")
                return None
        return None

    def get_det_gt(self, image):
        if "det" in self.anchor_model_id:
            try:
                # 同一张图复用 get_det_embed 那次推理的 head 输出，不再重跑 DBNet
                prob_map = self._get_det_outputs(image)['head']
                if prob_map is not None:
                    # 下采样到 1/4
                    prob_map_small = F.interpolate(prob_map, scale_factor=0.25, mode='bilinear', align_corners=False)
                    return prob_map_small
            except Exception as e:
                print(f"[Error in get_det_gt] {e}")
                return None
        return None

    # === 通用解码逻辑 ===
    def decode_det_tokens(self, tokens, det_feats):
        batch_size, n_tokens, c = tokens.shape
        b, c_feat, h, w = det_feats.shape
        feat_flat = det_feats.view(b, c_feat, -1)
        attn_map = torch.bmm(tokens, feat_flat)
        attn_map = attn_map.view(b, n_tokens, h, w)
        # 聚合各 token 的结果 -> [B, 1, H, W]
        pred_map = torch.mean(attn_map, dim=1, keepdim=True)

        # 必须有 Sigmoid 确保输出在 [0, 1]
        pred_map = torch.sigmoid(pred_map)

        return pred_map

    def _get_layout_outputs(self, image):
        """
        单张图只执行一次 DocLayout-YOLO 推理（统一 1024x1024 方形输入），
        同时产出 neck 特征与检测结果，供 layout/flow 的 embed 与 GT 四处复用。
        """
        entry = self._cache_get(self._layout_cache, image)
        if entry is not None:
            return entry
        if self._load_teacher_cache(image):
            return self._cache_get(self._layout_cache, image)

        # 强制 Resize 到 1024x1024 正方形，忽略长宽比，
        # 保证输出 Feature Map 尺寸固定为 [1, C, 32, 32]，可在 Batch 维度 cat
        target_size = (1024, 1024)
        img_input = self._to_pil(image).resize(target_size)

        with torch.no_grad():
            results = self.layout_model.predict(img_input, imgsz=1024, device=self.device, verbose=False, save=False)

        # 从 Hook 获取特征，防御性尺寸对齐
        feat = self.layout_features.get('neck', None)
        if feat is not None:
            expected_shape = (32, 32)
            if feat.shape[-2:] != expected_shape:
                feat = F.interpolate(feat, size=expected_shape, mode='bilinear', align_corners=False)
            feat = feat.detach()

        outputs = {'feat': feat, 'result': results[0]}
        self._cache_put(self._layout_cache, image, outputs)
        return outputs

    def get_layout_embed(self, image):
        if "layout" in self.anchor_model_id or "flow" in self.anchor_model_id:
            return self._get_layout_outputs(image)['feat']
        return None

    def get_layout_gt(self, image):
        if "layout" in self.anchor_model_id:
            # 复用与 layout 特征同一次 YOLO 推理的检测结果
            det_res = self._get_layout_outputs(image)['result']

            # 统一输出 128x128 的监督信号，节省显存
            target_h, target_w = 128, 128
            gt_map = torch.zeros((1, target_h, target_w), device=self.device, dtype=torch.float32)

            if det_res.boxes is not None:
                # 获取归一化坐标 xyxyn (0-1)
                boxes = det_res.boxes.xyxyn
                cls = det_res.boxes.cls

                # 光栅化循环
                for box, c in zip(boxes, cls):
                    c_idx = int(c.item())
                    # 'abandon' (2) 或未知类别置 0
                    if c_idx == 2: continue
                    val = self.layout_class_vals.get(c_idx, 0.2)

                    # 映射坐标
                    x1 = int(box[0] * target_w)
                    y1 = int(box[1] * target_h)
                    x2 = int(box[2] * target_w)
                    y2 = int(box[3] * target_h)

                    # 边界截断
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(target_w, x2), min(target_h, y2)

                    if x2 > x1 and y2 > y1:
                        # 重叠区域取 max，保留更高重要度的语义（避免后绘制的低值框抹掉高值区域）
                        gt_map[:, y1:y2, x1:x2] = gt_map[:, y1:y2, x1:x2].clamp(min=val)

            return gt_map.unsqueeze(0) # [1, 1, 128, 128]
        return None

    def _get_coord_channels(self, H, W, device, dtype):
        """8 个位置通道：x/y 线性项（表达单调顺序场）+ Fourier 项（表达分栏/周期结构），
        y 方向多一组频率（阅读顺序以自上而下为主）。按尺寸缓存。"""
        key = (H, W)
        if getattr(self, '_coord_cache_key', None) != key:
            ys = torch.linspace(0, 1, H)
            xs = torch.linspace(0, 1, W)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            chans = [
                xx, yy,
                torch.sin(2 * math.pi * xx), torch.cos(2 * math.pi * xx),
                torch.sin(2 * math.pi * yy), torch.cos(2 * math.pi * yy),
                torch.sin(4 * math.pi * yy), torch.cos(4 * math.pi * yy),
            ]
            self._coord_cache = torch.stack(chans, dim=0).unsqueeze(0)  # [1,8,H,W]
            self._coord_cache_key = key
        return self._coord_cache.to(device=device, dtype=dtype)

    def get_flow_decode_feats(self, image):
        """flow 解码特征 = DBNet neck [1,256,128,128] + 8 个坐标通道 → [1,264,128,128]
        （DBNet 特征已在 det 缓存中，零额外推理）"""
        if "flow" not in self.anchor_model_id:
            return None
        if "det" not in self.anchor_model_id or self.dbnet is None:
            print("[Warn] flow 解码需要 det (DBNet) 特征，当前未启用，flow 监督跳过")
            return None
        feat = self._get_det_outputs(image)['neck']  # [1,256,H,W]
        if feat is None:
            return None
        B, C, H, W = feat.shape
        coord = self._get_coord_channels(H, W, feat.device, feat.dtype)
        return torch.cat([feat, coord.expand(B, -1, -1, -1)], dim=1)

    def get_flow_embed(self, image):
        return self.get_flow_decode_feats(image)

    def _extract_word_boxes(self, image):
        """从 DBNet head 概率图（det 缓存，零额外推理）提取词级框 (xyxyn)，
        作为 LayoutReader 的原生粒度输入（ReadingBank 为词级训练）"""
        if "det" not in self.anchor_model_id or self.dbnet is None:
            return []
        head = self._get_det_outputs(image)['head']
        if head is None:
            return []
        prob = head[0, 0].float().cpu().numpy()
        H, W = prob.shape
        binary = (prob > 0.3).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 15 or h > 0.15 * H:  # 过滤噪点与非文本大块
                continue
            boxes.append([x / W, y / H, (x + w) / W, (y + h) / H])
        if len(boxes) > 500:
            boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:500]
        return boxes

    def _get_reading_order(self, image, outputs, boxes):
        """获取块级阅读顺序并缓存，同图只算一次。
        主路径（实证块序 vs y 相关 0.97）：词级 LayoutReader + 聚合到块——
        LayoutReader 在 ReadingBank 词级框上训练，直接输入 5~20 个块级大框严重分布外，
        输出近乎随机（实测 vs y 仅 0.23）；回退链：块级 LayoutReader → 启发式。"""
        if outputs.get('order') is None:
            order = None
            words = self._extract_word_boxes(image)
            if self.layoutreader is not None and len(words) >= 10:
                img_w, img_h = self._to_pil(image).size
                worder = layoutreader_predict_reading_order(self.layoutreader, words, img_w, img_h)
                wc = np.array(words)
                ranks = np.empty(len(words)); ranks[:] = np.nan
                for r, wi in enumerate(worder):
                    ranks[wi] = r
                wcx = (wc[:, 0] + wc[:, 2]) / 2
                wcy = (wc[:, 1] + wc[:, 3]) / 2
                # 块分数 = 块内词 rank 均值；无词的块用 y 中心映射到 rank 量纲插入
                scores = []
                for x1, y1, x2, y2 in boxes:
                    m = (wcx >= x1) & (wcx <= x2) & (wcy >= y1) & (wcy <= y2)
                    if m.sum() > 0:
                        scores.append(float(np.nanmean(ranks[m])))
                    else:
                        scores.append(float((y1 + y2) / 2 * len(words)))
                order = sorted(range(len(boxes)), key=lambda i: scores[i])
            elif self.layoutreader is not None:
                # 词框不足（纯图表页等）：回退块级输入
                img_w, img_h = self._to_pil(image).size
                order = layoutreader_predict_reading_order(self.layoutreader, boxes, img_w, img_h)
            if order is None:
                order = get_reading_order_indices(boxes, y_threshold=0.02)
            outputs['order'] = order
        return outputs['order']

    def get_flow_supervision(self, image):
        """
        flow 监督信号：
        返回 (剪影GT [1,1,128,128], 按阅读顺序排列的 xyxyn 框列表)
        - 剪影 GT 只编码"框在哪"（框内 1），不把顺序压缩成灰度
        - 顺序以框列表的排列传递，由 loss 的 ranking 项直接监督
        - 检测框/顺序与 layout 特征同源（同一次 1024 推理 + 缓存）
        """
        if "flow" not in self.anchor_model_id:
            return None, None
        outputs = self._get_layout_outputs(image)
        result = outputs['result']

        target_h, target_w = 128, 128
        sil = torch.zeros((1, 1, target_h, target_w), device=self.device, dtype=torch.float32)
        boxes_in_order = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxyn.cpu().numpy().tolist()
            order = self._get_reading_order(image, outputs, boxes)
            boxes_in_order = [boxes[i] for i in order]
            for x1, y1, x2, y2 in boxes:
                xs, ys = max(0, int(x1 * target_w)), max(0, int(y1 * target_h))
                xe, ye = min(target_w, int(x2 * target_w)), min(target_h, int(y2 * target_h))
                if xe > xs and ye > ys:
                    sil[:, :, ys:ye, xs:xe] = 1.0

        return sil, boxes_in_order
