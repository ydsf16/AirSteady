import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import sys
import torch
from dataclasses import dataclass
from utils import get_airsteady_cache_dir
from typing import List
# from scipy.optimize import least_squares
from typing import Tuple
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def resource_path(relative_path: str) -> str:
    """兼容普通运行和 PyInstaller 打包后的资源路径."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller 打包后一切会被解压到这个临时目录
        base_path = sys._MEIPASS
    else:
        # 开发时：以当前文件所在目录为基准（即 core/）
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

@dataclass
class TrackFrame:
    timestamp: float
    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    valid: bool
    num_obj: int

class TrackEngine:
    """
    AirSteady 跟踪算法：
    """  
    def __init__(
        self,
        model_path: str = "model.pt"
    ):
        # 如果没有显式传，就用 core/model/model.pt
        if not os.path.isabs(model_path):
            # 注意这里拼的是 model/<你的文件名>，匹配 core/model 目录
            model_path = resource_path(os.path.join("model", model_path))

        self.model_path = model_path
        self.tracker_cfg = resource_path(os.path.join("model", "tracker.yaml"))

        # YOLO 模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 视频相关
        self.cap = None
        self.video_path = None
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.scale_width = 0
        self.scale_height = 0
        self.scale_ratio = 1.0
        
        self.total_frames = 0
        self.current_frame_idx = 0

        self.has_video = False
        self.names = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

        # Limit max image size in preview!!!
        self.max_width = 600
        self.max_height = 600

        # Track result.
        self.track_results = []

        # 可选：做一次 warm-up，避免第一帧卡得特别明显
        print("Loading model")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            fake_frame,
            device=self.device,
            verbose=False,
        )
        print("Model loaded")

        self.tmp_video_path = os.path.join(get_airsteady_cache_dir(), "tmp_track.mp4")
        
        self.temp_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
    
    def open_video(self, path: str):
        """打开视频并初始化尺寸等信息"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap = cap
        self.video_path = path
        self.fps = float(fps)
        self.total_frames = total_frames
        self.width = w
        self.height = h
        self.current_frame_idx = 0

        self.scale_ratio = 1.0
        if self.width > self.max_width:
            print("1")
            self.scale_ratio = min(self.scale_ratio, float(self.max_width) / self.width )
        if (self.height > self.max_height):
            print("2")
            self.scale_ratio = min(self.scale_ratio, float(self.max_height) / self.height )
    
        # 裁剪窗口尺寸
        self.scale_width = int(self.width * self.scale_ratio)
        self.scale_height = int(self.height * self.scale_ratio)

        print("Scale ratio", self.scale_ratio)
        print("scale_width", self.scale_width)
        print("scale_height", self.scale_height)

        # Clear track results.
        self.track_results.clear()

        self.temp_writer = cv2.VideoWriter(self.tmp_video_path, self.fourcc, self.fps, (self.scale_width, self.scale_height))

    def finished(self):
        if self.temp_writer is not None:
            self.temp_writer.release()
            self.temp_writer = None
            print("finished track writter")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("finished track cap")

    # ------------------------------------------------------------------
    # 单步前向：读取下一帧，做 YOLO + 跟踪 + 稳像（在线预览）
    # ------------------------------------------------------------------
    def next_frame(self, target_class = "airplane"):
        """
        读取下一帧并处理（用于在线预览）。
        """
        if self.cap is None:
            return None, None

        success, frame = self.cap.read()
        if not success:
            return None, None
        
        # scale images here.
        if self.scale_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (self.scale_width, self.scale_height),  # 注意是 (width, height)
                interpolation=cv2.INTER_AREA,          # 下采样推荐用 INTER_AREA
            )
        
        if self.temp_writer is not None:
            self.temp_writer.write(frame)

        # class name -> class id.
        self.target_class_name = target_class
        self.target_class_id = None
        for i, name in self.names.items():
            if name == self.target_class_name:
                self.target_class_id = int(i)
                break

        # 当前帧索引
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # YOLO 跟踪（persist=True 保留 track id）
        result = self.model.track(
            frame,
            persist=True,
            verbose=False,
            device=self.device,
            tracker=self.tracker_cfg,
        )[0]

        # 用原始 frame 拷贝作为可视化底图
        vis_frame = frame.copy()

        boxes = result.boxes
        selected_xywh: list[tuple[float, float, float, float]] = []
        selected_confs: list[float] = []

        # ---------- STEP1: 提取指定类别的 bbox + conf ----------
        if boxes is not None and boxes.xywh is not None:
            # xywh: [cx, cy, w, h] in pixels
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()

            for (cx, cy, w_box, h_box), conf, cls_id in zip(xywh, confs, cls_ids):
                if cls_id == self.target_class_id:
                    selected_xywh.append((float(cx), float(cy), float(w_box), float(h_box)))
                    selected_confs.append(float(conf))

        num_obj = len(selected_xywh)

        # ---------- STEP2: 绘制角标 + 中间十字 ----------
        # 多目标时轮询颜色
        colors = [
            (0, 255, 0),      # 绿
            (0, 255, 255),    # 黄
            (255, 0, 0),      # 蓝（BGR）
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青
        ]
        line_thickness = 4
        for idx, (cx, cy, w_box, h_box) in enumerate(selected_xywh):
            color = colors[idx % len(colors)]

            x1 = int(cx - w_box / 2.0)
            y1 = int(cy - h_box / 2.0)
            x2 = int(cx + w_box / 2.0)
            y2 = int(cy + h_box / 2.0)

            w_i = max(1, x2 - x1)
            h_i = max(1, y2 - y1)
            corner_len = max(1, int(0.2 * min(w_i, h_i)))   # 角线长度
            cross_len = max(1, int(0.15 * min(w_i, h_i)))   # 十字长度

            # 左上角
            cv2.line(vis_frame, (x1, y1), (x1 + corner_len, y1), color, line_thickness)
            cv2.line(vis_frame, (x1, y1), (x1, y1 + corner_len), color, line_thickness)

            # 右上角
            cv2.line(vis_frame, (x2, y1), (x2 - corner_len, y1), color, line_thickness)
            cv2.line(vis_frame, (x2, y1), (x2, y1 + corner_len), color, line_thickness)

            # 左下角
            cv2.line(vis_frame, (x1, y2), (x1 + corner_len, y2), color, line_thickness)
            cv2.line(vis_frame, (x1, y2), (x1, y2 - corner_len), color, line_thickness)

            # 右下角
            cv2.line(vis_frame, (x2, y2), (x2 - corner_len, y2), color, line_thickness)
            cv2.line(vis_frame, (x2, y2), (x2, y2 - corner_len), color, line_thickness)

            # 中心十字
            cx_i = int(cx)
            cy_i = int(cy)
            cv2.line(vis_frame, (cx_i - cross_len, cy_i), (cx_i + cross_len, cy_i), color, line_thickness)
            cv2.line(vis_frame, (cx_i, cy_i - cross_len), (cx_i, cy_i + cross_len), color, line_thickness)

        # ---------- STEP3: 组织 TrackFrame ----------
        # 时间戳：用“当前帧在视频中的时间”
        if hasattr(self, "fps") and self.fps > 1e-6 and hasattr(self, "current_frame_idx"):
            timestamp = float(self.current_frame_idx / self.fps)
        else:
            timestamp = 0.0

        if num_obj == 1:
            # 只有一个目标 → valid=True
            cx, cy, w_box, h_box = selected_xywh[0]
            conf = selected_confs[0]
            track_frame = TrackFrame(
                timestamp=timestamp,
                frame_idx=self.current_frame_idx,
                cx=cx,
                cy=cy,
                w=w_box,
                h=h_box,
                conf=conf,
                valid=True,
                num_obj=num_obj,
            )
        else:
            # 0 个 或 >1 个目标 → 统统 invalid
            track_frame = TrackFrame(
                timestamp=timestamp,
                frame_idx=self.current_frame_idx,
                cx=0.0,
                cy=0.0,
                w=0.0,
                h=0.0,
                conf=0.0,
                valid=False,
                num_obj=num_obj,
            )

        self.track_results.append(track_frame)
        return vis_frame, track_frame
    



@dataclass
class CropFrame:
    timestamp: float
    frame_idx: int
    crop_center_x: float
    crop_center_y: float
    crop_width: float
    crop_height: float
    scale: float # 是否对裁切参数缩放，目的是用于映射到大尺寸图像上的裁切
    clamp: bool  # 是否触碰了clamp，用于提示用户，增大裁切。


def build_observations_and_weights(
    track_result: List[TrackFrame],
    img_width: int,
    img_height: int,
    max_crop_ratio: float,
    density_window_sec: float = 2.0,      # 时间窗口
    density_min_ratio: float = 0.3,       # 最小观测占比，比如 10%
    long_gap_sec: float = 5.0,          # 回中阈值，秒
    recenter_weight_scale: float = 0.01, # 回中观测权重相对比例
):
    """
    Step1 + 权重构造 + 观测密度/长时间丢失处理。
    """

    n = len(track_result)
    W = float(img_width)
    H = float(img_height)
    max_crop_ratio = float(np.clip(max_crop_ratio, 0.0, 0.5))

    timestamps = np.array([f.timestamp for f in track_result], dtype=float)
    frame_idx = np.array([f.frame_idx for f in track_result], dtype=int)
    cx_raw = np.array([f.cx for f in track_result], dtype=float)
    cy_raw = np.array([f.cy for f in track_result], dtype=float)
    w_raw = np.array([f.w for f in track_result], dtype=float)
    h_raw = np.array([f.h for f in track_result], dtype=float)
    conf_raw = np.array([f.conf for f in track_result], dtype=float)
    valid_raw = np.array([f.valid for f in track_result], dtype=bool)
    num_obj_raw = np.array([f.num_obj for f in track_result], dtype=int)

    # ---- Step1: max_crop_ratio 下的 clamp（几何上可行区） ----
    half_w_max = W * (1.0 - max_crop_ratio) / 2.0
    half_h_max = H * (1.0 - max_crop_ratio) / 2.0
    min_cx_all = half_w_max
    max_cx_all = W - half_w_max
    min_cy_all = half_h_max
    max_cy_all = H - half_h_max

    obs_cx = cx_raw.copy()
    obs_cy = cy_raw.copy()
    obs_cx[~valid_raw] = W / 2.0
    obs_cy[~valid_raw] = H / 2.0

    idx_valid = np.where(valid_raw)[0]
    if idx_valid.size > 0:
        obs_cx[idx_valid] = np.clip(obs_cx[idx_valid], min_cx_all, max_cx_all)
        obs_cy[idx_valid] = np.clip(obs_cy[idx_valid], min_cy_all, max_cy_all)

    # ---- 规则 1~4: 构造“原始的真实观测 mask” real_effective_mask ----
    real_mask = valid_raw.copy()
    real_mask &= (num_obj_raw == 1)

    # 边缘贴边 & 过大过滤（用你刚才讨论的边缘规则比较好）
    margin_left   = cx_raw - 0.5 * w_raw
    margin_right  = W - (cx_raw + 0.5 * w_raw)
    margin_top    = cy_raw - 0.5 * h_raw
    margin_bottom = H - (cy_raw + 0.5 * h_raw)
    min_margin_x = np.minimum(margin_left, margin_right)
    min_margin_y = np.minimum(margin_top, margin_bottom)
    edge_thresh_x = 0.05 * W
    edge_thresh_y = 0.05 * H
    bbox_at_edge = (min_margin_x < edge_thresh_x) | (min_margin_y < edge_thresh_y)
    bbox_too_large = (w_raw > 0.95 * W) | (h_raw > 0.95 * H)
    bad_bbox = bbox_at_edge | bbox_too_large

    real_mask &= ~bad_bbox

    # 基础权重（还没做密度与 Huber）
    conf_norm = np.clip(conf_raw, 0.0, 1.0)
    base_weights = np.zeros(n, dtype=float)
    base_weights[real_mask] = conf_norm[real_mask]

    # -------- A) 观测密度筛选：按照比例判断 --------
    if n > 0 and density_window_sec > 0.0 and density_min_ratio > 0.0:
        # 估算 fps
        if timestamps[-1] > timestamps[0]:
            fps_est = (n - 1) / (timestamps[-1] - timestamps[0])
        else:
            fps_est = 30.0

        window_frames = max(1, int(density_window_sec * fps_est))

        # 卷积统计窗口内“真实观测帧数”
        kernel = np.ones(window_frames, dtype=float)
        density = np.convolve(real_mask.astype(float), kernel, mode="same")

        # 由比例换算成“需要的最小数量”
        density_min_ratio = float(np.clip(density_min_ratio, 0.0, 1.0))
        min_count = max(1, int(round(density_min_ratio * window_frames)))

        low_density = density < float(min_count)

        base_weights[low_density] = 0.0
        real_mask[low_density] = False

    # -------- B) 长时间完全无观测：回中处理 --------
    # 这里使用 real_mask（经过密度筛选后的）来定义“无观测”
    if n > 0 and long_gap_sec > 0.0 and recenter_weight_scale > 0.0:
        # 先找出真实观测的全局平均权重，给回中点一个小比例
        nonzero_w = base_weights[base_weights > 0]
        if nonzero_w.size > 0:
            avg_w = float(nonzero_w.mean())
        else:
            avg_w = 1.0
        recenter_w = avg_w * recenter_weight_scale

        i = 0
        while i < n:
            if real_mask[i]:
                i += 1
                continue
            # 找一个连续的 real_mask==False 段 [start, end)
            start = i
            while i < n and not real_mask[i]:
                i += 1
            end = i  # [start, end)

            # 计算这段时间长度
            dur = timestamps[end - 1] - timestamps[start]
            if dur >= long_gap_sec:
                # 在这一整段内做“回中”
                obs_cx[start:end] = W / 2.0
                obs_cy[start:end] = H / 2.0
                base_weights[start:end] = recenter_w
                # 注意：这里我们不把 real_mask 改成 True
                # real_mask 仍然表示“来自真实 tracking 的观测”
                # 回中点只参与 BA，不用于裁切比例统计

    # 最终 solver 使用的权重（包括回中点）
    weights = base_weights.copy()
    solver_effective_mask = weights > 0

    # 如果所有 weights 都是 0，退化成简单平滑
    if np.all(weights == 0):
        weights[:] = 1.0
        solver_effective_mask[:] = True

    return {
        "timestamps": timestamps,
        "frame_idx": frame_idx,
        "cx_raw": cx_raw,
        "cy_raw": cy_raw,
        "obs_cx": obs_cx,
        "obs_cy": obs_cy,
        "weights": weights,                   # 给 solver / Huber 用
        "real_effective_mask": real_mask,     # 真正 tracking 好的帧
        "solver_effective_mask": solver_effective_mask,
        "W": W,
        "H": H,
    }


def _solve_axis_sparse_linear(
    obs: np.ndarray,
    weights: np.ndarray,
    smooth_factor: float,
) -> np.ndarray:
    """
    只做纯 L2 最小二乘的单轴平滑（无鲁棒核）。
    """
    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return obs.copy()

    smooth_factor = float(np.clip(smooth_factor, 0.0, 1.0))
    lambda_pos = 10.0 * (smooth_factor ** 2)
    lambda_vel = 5.0 * (smooth_factor ** 2)

    diag0 = np.zeros(n, dtype=float)       # A_ii
    diag1 = np.zeros(n - 1, dtype=float)   # A_{i,i+1} / A_{i+1,i}
    diag2 = np.zeros(n - 2, dtype=float)   # A_{i,i+2} / A_{i+2,i}

    b = np.zeros(n, dtype=float)

    # --- 数据项 ---
    diag0 += weights
    b += weights * obs

    # --- 位置平滑 ---
    if lambda_pos > 0.0:
        for i in range(1, n):
            diag0[i] += lambda_pos
            diag0[i - 1] += lambda_pos
            diag1[i - 1] += -lambda_pos

    # --- 速度平滑 ---
    if lambda_vel > 0.0:
        for i in range(2, n):
            diag0[i] += lambda_vel
            diag0[i - 1] += 4.0 * lambda_vel
            diag0[i - 2] += lambda_vel

            diag1[i - 1] += -2.0 * lambda_vel
            diag1[i - 2] += -2.0 * lambda_vel

            diag2[i - 2] += lambda_vel

    diagonals = [diag2, diag1, diag0, diag1, diag2]
    offsets = [-2, -1, 0, 1, 2]
    A = diags(diagonals, offsets, shape=(n, n), format="csc")

    p = spsolve(A, b)
    return np.asarray(p, dtype=float)

def _solve_axis_sparse_huber(
    obs: np.ndarray,
    base_weights: np.ndarray,
    smooth_factor: float,
    huber_delta: float = 10.0,
    max_iter: int = 5,
) -> np.ndarray:
    """
    对单个轴做 Huber-IRLS 平滑：
    - base_weights: 原始置信度 (conf / 0/1 mask)，无鲁棒核
    - huber_delta: 残差阈值（像素），大于该值的残差权重会衰减
    """

    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return obs.copy()

    # 确保权重非负
    base_w = np.clip(base_weights, 0.0, np.inf)

    # 初始解：用纯 L2 做一遍
    w_cur = base_w.copy()
    p = _solve_axis_sparse_linear(obs, w_cur, smooth_factor)

    for it in range(max_iter):
        # 残差
        r = p - obs
        abs_r = np.abs(r)

        # Huber 权重：psi(r)/r
        # psi(r) = r (|r|<=delta), = delta * sign(r) (|r|>delta)
        # => psi(r)/r = 1 (|r|<=delta), = delta/|r| (|r|>delta)
        robust_factor = np.ones_like(abs_r, dtype=float)

        mask = abs_r > 1e-6
        big = mask & (abs_r > huber_delta)
        # 对大残差，权重缩小为 delta / |r|
        robust_factor[big] = huber_delta / abs_r[big]

        # 合成新的权重：base_w * robust_factor
        w_cur = base_w * robust_factor

        # 防止所有点都被降到 0
        if np.all(w_cur == 0):
            # 退化成纯平滑：给所有点一个小权重
            w_cur[:] = 1.0

        # 再跑一次线性 L2 解
        p = _solve_axis_sparse_linear(obs, w_cur, smooth_factor)

    return p


def _solve_axis_sparse(
    obs: np.ndarray,
    weights: np.ndarray,
    smooth_factor: float,
) -> np.ndarray:
    """
    对单个轴（x 或 y）做全局平滑：
    最小化 sum_i w_i (p_i - obs_i)^2
           + λ_pos sum_i (p_i - p_{i-1})^2
           + λ_vel sum_i (p_i - 2p_{i-1} + p_{i-2})^2

    使用五对角稀疏矩阵 + spsolve，复杂度约 O(N)。
    """
    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        # 帧太少，直接用观测
        return obs.copy()

    smooth_factor = float(np.clip(smooth_factor, 0.0, 1.0))

    # 平滑强度，可以按需要调一下系数
    lambda_pos = 10.0 * (smooth_factor ** 2)
    lambda_vel = 5.0 * (smooth_factor ** 2)

    # 主对角线 + 上一对角线 + 上二对角线
    diag0 = np.zeros(n, dtype=float)       # 对应 i,i
    diag1 = np.zeros(n - 1, dtype=float)   # 对应 i,i+1
    diag2 = np.zeros(n - 2, dtype=float)   # 对应 i,i+2

    # 右侧向量 b
    b = np.zeros(n, dtype=float)

    # ---------- 1) 数据项：w_i (p_i - obs_i)^2 ----------
    # 展开：w_i p_i^2 - 2 w_i obs_i p_i + const
    # => A_ii += w_i, b_i += w_i * obs_i
    diag0 += weights
    b += weights * obs

    # ---------- 2) 位置平滑项：λ_pos (p_i - p_{i-1})^2 ----------
    # 对 i=1..n-1
    # 展开：λ(p_i^2 - 2 p_i p_{i-1} + p_{i-1}^2)
    # => A_ii += λ, A_{i-1,i-1} += λ, A_{i,i-1} = A_{i-1,i} += -λ
    if lambda_pos > 0.0:
        for i in range(1, n):
            diag0[i] += lambda_pos
            diag0[i - 1] += lambda_pos
            # A_{i-1, i} 和 A_{i, i-1} 都在 diag1 上体现一次
            diag1[i - 1] += -lambda_pos

    # ---------- 3) 速度平滑项：λ_vel (p_i - 2p_{i-1} + p_{i-2})^2 ----------
    # 对 i=2..n-1，记 a=p_i, b=p_{i-1}, c=p_{i-2}
    # (a - 2b + c)^2 = a^2 + 4b^2 + c^2 -4ab + 2ac -4bc
    # => 对应贡献：
    #    A_ii       += λ
    #    A_{i-1,i-1}+= 4λ
    #    A_{i-2,i-2}+= λ
    #    A_{i,i-1}  = A_{i-1,i}   += -2λ  （因为 2A_{i,i-1} = -4λ）
    #    A_{i,i-2}  = A_{i-2,i}   +=  λ   （因 2A_{i,i-2} =  2λ）
    #    A_{i-1,i-2}= A_{i-2,i-1} += -2λ  （因 2A_{i-1,i-2} = -4λ）
    if lambda_vel > 0.0:
        for i in range(2, n):
            # 对角线
            diag0[i] += lambda_vel
            diag0[i - 1] += 4.0 * lambda_vel
            diag0[i - 2] += lambda_vel

            # 相邻 (i, i-1)
            diag1[i - 1] += -2.0 * lambda_vel

            # 相邻 (i-1, i-2)
            diag1[i - 2] += -2.0 * lambda_vel

            # 间隔一位 (i, i-2)
            diag2[i - 2] += lambda_vel

    # ---------- 构造稀疏五对角矩阵 ----------
    # 对称矩阵 A：对角线 offset [-2, -1, 0, 1, 2]
    # 上下对角线共享同一数组 (对称)
    diagonals = [
        diag2,          # offset = -2
        diag1,          # offset = -1
        diag0,          # offset = 0
        diag1,          # offset = +1
        diag2,          # offset = +2
    ]
    offsets = [-2, -1, 0, 1, 2]

    A = diags(diagonals, offsets, shape=(n, n), format="csc")

    # ---------- 线性求解 A p = b ----------
    p = spsolve(A, b)    # 返回 np.ndarray

    return np.asarray(p, dtype=float)

def solve_smooth_trajectory(
    obs_cx: np.ndarray,
    obs_cy: np.ndarray,
    weights: np.ndarray,
    smooth_factor: float,
    huber_delta: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Huber-IRLS 版本的双轴平滑。
    当 smooth_factor 非常小时，直接返回观测，避免线性系统奇异。
    """
    n = len(obs_cx)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    # 如果平滑度几乎为 0，就不要建方程，直接用观测（已经过 clamp + 回中处理）
    if smooth_factor <= 1e-6:
        px = np.asarray(obs_cx, dtype=float).copy()
        py = np.asarray(obs_cy, dtype=float).copy()
        return px, py

    px = _solve_axis_sparse_huber(
        obs=obs_cx,
        base_weights=weights,
        smooth_factor=smooth_factor,
        huber_delta=huber_delta,
        max_iter=5,
    )
    py = _solve_axis_sparse_huber(
        obs=obs_cy,
        base_weights=weights,
        smooth_factor=smooth_factor,
        huber_delta=huber_delta,
        max_iter=5,
    )
    return px, py


def compute_global_crop_ratio(
    px: np.ndarray,
    py: np.ndarray,
    W: float,
    H: float,
    max_crop_ratio: float,
    effective_mask: np.ndarray,
) -> float:
    """
    Step2: 只用有效帧(effective_mask==True)来计算全局裁切比例。
    - max_crop_ratio: 最大允许裁切比例（0 表示完全不裁切）
    """
    # 限制到安全范围
    max_crop_ratio = float(np.clip(max_crop_ratio, 0.0, 0.9))

    if max_crop_ratio <= 1e-6:
        # 明确表示“不裁切”
        return 0.0

    if np.any(effective_mask):
        px_eff = px[effective_mask]
        py_eff = py[effective_mask]
    else:
        px_eff = px
        py_eff = py

    # 如果 px/py 里有 NaN，这里直接防御
    valid = np.isfinite(px_eff) & np.isfinite(py_eff)
    if not np.any(valid):
        # 全部 NaN 的话，退化成不裁切
        return 0.0

    px_eff = px_eff[valid]
    py_eff = py_eff[valid]

    dx_left = px_eff
    dx_right = W - px_eff
    dy_top = py_eff
    dy_bottom = H - py_eff

    min_margin_x = np.minimum(dx_left, dx_right)
    min_margin_y = np.minimum(dy_top, dy_bottom)

    crop_ratio_x = 1.0 - 2.0 * min_margin_x / (W + 1e-6)
    crop_ratio_y = 1.0 - 2.0 * min_margin_y / (H + 1e-6)

    per_frame_required = np.maximum(crop_ratio_x, crop_ratio_y)
    per_frame_required = np.clip(per_frame_required, 0.0, 1.0)

    # 再防一个：如果这里意外出现 NaN
    valid2 = np.isfinite(per_frame_required)
    if not np.any(valid2):
        required_crop_ratio = 0.0
    else:
        required_crop_ratio = float(np.max(per_frame_required[valid2]))

    crop_ratio = float(min(required_crop_ratio, max_crop_ratio))
    crop_ratio = float(np.clip(crop_ratio, 0.0, 1.0))
    return crop_ratio


def build_crop_frames(
    timestamps: np.ndarray,
    frame_idx: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    W: float,
    H: float,
    crop_ratio: float,
    effective_mask: np.ndarray,
) -> List[CropFrame]:
    """
    Step4: 用平滑后的 px, py + 全局 crop_ratio 生成 list[CropFrame]。
    - 所有帧都会输出 CropFrame
    - 但 clamp 标记只对 effective_mask==True 的帧有效
    """

    n = len(px)
    crop_width = W * (1.0 - crop_ratio)
    crop_height = H * (1.0 - crop_ratio)
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0

    min_cx = half_w
    max_cx = W - half_w
    min_cy = half_h
    max_cy = H - half_h

    out: List[CropFrame] = []

    for i in range(n):
        x = px[i]
        y = py[i]
        clamped = False

        # 所有帧都要保证裁切框不出界，所以都要 clamp。
        # 但 clamp 标记只对有效帧生效（无效帧不提示）。
        if x < min_cx:
            x = min_cx
            if effective_mask[i]:
                clamped = True
        elif x > max_cx:
            x = max_cx
            if effective_mask[i]:
                clamped = True

        if y < min_cy:
            y = min_cy
            if effective_mask[i]:
                clamped = True
        elif y > max_cy:
            y = max_cy
            if effective_mask[i]:
                clamped = True

        out.append(
            CropFrame(
                timestamp=float(timestamps[i]),
                frame_idx=int(frame_idx[i]),
                crop_center_x=float(x),
                crop_center_y=float(y),
                crop_width=float(crop_width),
                crop_height=float(crop_height),
                scale=1.0,  # 在工作分辨率下，外层可用 orig_W / W 做映射
                clamp=clamped,
            )
        )

    return out

def plot_planning_debug(
    timestamps: np.ndarray,
    cx_raw: np.ndarray,
    cy_raw: np.ndarray,
    obs_cx: np.ndarray,
    obs_cy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    effective_mask: np.ndarray,
    W: float,
    H: float,
    crop_ratio: float,
):
    """
    Debug 可视化：
    - 横轴：时间戳
    - X 子图：中心 x 轨迹 + 裁切左右边界
    - Y 子图：中心 y 轨迹 + 裁切上下边界
    - 纵轴范围固定为 [0, W] / [0, H]，与图像坐标一致
    """

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[planning debug] matplotlib not available: {e}")
        return

    t = timestamps
    eff = effective_mask
    crop_ratio = float(np.clip(crop_ratio, 0.0, 1.0))

    # 基于最终 crop_ratio 计算裁切宽高
    crop_width = W * (1.0 - crop_ratio)
    crop_height = H * (1.0 - crop_ratio)
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0

    # 对每一帧计算裁切框的边界（并 clamp 到图像范围内）
    left_x = np.clip(px - half_w, 0.0, W)
    right_x = np.clip(px + half_w, 0.0, W)

    top_y = np.clip(py - half_h, 0.0, H)
    bottom_y = np.clip(py + half_h, 0.0, H)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ---------------- X 方向 ----------------
    ax = axes[0]
    ax.set_title("Crop Planning Debug - X (center & crop bounds)")
    # 原始 cx
    ax.plot(t, cx_raw, "k.", alpha=0.3, label="raw cx")
    # clamp 后观测
    ax.plot(t, obs_cx, "b-", alpha=0.6, label="obs cx (clamped if valid)")
    ax.plot(t[eff], obs_cx[eff], "bo", alpha=0.9, label="effective obs")
    # 平滑后的中心
    ax.plot(t, px, "r-", alpha=0.9, label="smoothed cx")
    # 裁切左右边界（淡色曲线）
    ax.plot(t, left_x, linestyle="--", color="gray", alpha=0.5, label="crop left bound")
    ax.plot(t, right_x, linestyle="--", color="gray", alpha=0.5, label="crop right bound")
    # 图像边界线（0 和 W）
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.axhline(W, color="black", linewidth=0.5, alpha=0.5)

    ax.set_ylabel("cx (pixels)")
    ax.set_ylim(0.0, W)   # 范围与图像宽度一致
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ---------------- Y 方向 ----------------
    ay = axes[1]
    ay.set_title("Crop Planning Debug - Y (center & crop bounds)")
    # 原始 cy
    ay.plot(t, cy_raw, "k.", alpha=0.3, label="raw cy")
    # clamp 后观测
    ay.plot(t, obs_cy, "b-", alpha=0.6, label="obs cy (clamped if valid)")
    ay.plot(t[eff], obs_cy[eff], "bo", alpha=0.9, label="effective obs")
    # 平滑后的中心
    ay.plot(t, py, "r-", alpha=0.9, label="smoothed cy")
    # 裁切上下边界（淡色曲线）
    ay.plot(t, top_y, linestyle="--", color="gray", alpha=0.5, label="crop top bound")
    ay.plot(t, bottom_y, linestyle="--", color="gray", alpha=0.5, label="crop bottom bound")
    # 图像边界（0 和 H）
    ay.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ay.axhline(H, color="black", linewidth=0.5, alpha=0.5)

    ay.set_ylabel("cy (pixels)")
    ay.set_xlabel("time (s)")
    ay.set_ylim(0.0, H)   # 范围与图像高度一致
    ay.grid(True, alpha=0.3)
    ay.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def planning_crop_traj(
    track_result: List[TrackFrame],
    img_width: int,
    img_height: int,
    max_crop_ratio: float,
    smooth_factor: float = 0.5,
    debug: bool = False,
) -> List[CropFrame]:
    """
    总入口：根据跟踪结果规划运镜裁切轨迹。
    - track_result: list[TrackFrame]，长度 = 总帧数
    - img_width, img_height: 当前“工作分辨率”（TrackEngine 的 scale_width/height）
    - max_crop_ratio: 最大裁切比例（0~1，越大表示允许裁得越狠）
    - smooth_factor: 平滑度（0~1）
    - debug: 是否画 debug 曲线
    """
    if not track_result:
        return []

    # Step1: 观测 + 权重 + 各种 mask / 几何 clamp
    ctx = build_observations_and_weights(
        track_result=track_result,
        img_width=img_width,
        img_height=img_height,
        max_crop_ratio=max_crop_ratio,
    )

    timestamps = ctx["timestamps"]
    frame_idx = ctx["frame_idx"]
    cx_raw = ctx["cx_raw"]
    cy_raw = ctx["cy_raw"]
    obs_cx = ctx["obs_cx"]
    obs_cy = ctx["obs_cy"]
    weights = ctx["weights"]
    real_effective_mask = ctx["real_effective_mask"]      # 真正来自 tracking 的可靠观测
    # solver_effective_mask = ctx["solver_effective_mask"]  # 给 BA 用的（含回中点），目前不用单独用
    W = ctx["W"]
    H = ctx["H"]

    # Step3: Huber + 稀疏五对角平滑
    px, py = solve_smooth_trajectory(
        obs_cx=obs_cx,
        obs_cy=obs_cy,
        weights=weights,
        smooth_factor=smooth_factor,
        # huber_delta 你可以按需要调，这里先给个 10 像素
        huber_delta=10.0,
    )

    # Step2: 全局裁切比例 —— 只看“真实有效观测”的帧
    crop_ratio = compute_global_crop_ratio(
        px=px,
        py=py,
        W=W,
        H=H,
        max_crop_ratio=max_crop_ratio,
        effective_mask=real_effective_mask,
    )

    # Step4: 构造每一帧的 CropFrame，clamp flag 也只对真实观测生效
    crop_frames = build_crop_frames(
        timestamps=timestamps,
        frame_idx=frame_idx,
        px=px,
        py=py,
        W=W,
        H=H,
        crop_ratio=crop_ratio,
        effective_mask=real_effective_mask,
    )

    # Step5: Debug 可视化
    if debug:
        plot_planning_debug(
            timestamps=timestamps,
            cx_raw=cx_raw,
            cy_raw=cy_raw,
            obs_cx=obs_cx,
            obs_cy=obs_cy,
            px=px,
            py=py,
            effective_mask=real_effective_mask,
            W=W,
            H=H,
            crop_ratio=crop_ratio,
        )

    return crop_frames


## Crop
def _map_crop_to_full_res(
    frame_w: int,
    frame_h: int,
    work_w: int,
    work_h: int,
    cf: CropFrame,
):
    """
    将 CropFrame 中的中心/宽高，从工作分辨率 (work_w, work_h)
    映射到当前帧分辨率 (frame_w, frame_h)，并做 clamp.
    """
    # 按 x / y 分别算缩放因子（一般是等比的，但这里更稳妥）
    sx = frame_w / float(work_w)
    sy = frame_h / float(work_h)

    # 工作分辨率下的裁切参数 -> 放大到真实分辨率
    cx = cf.crop_center_x * sx
    cy = cf.crop_center_y * sy
    cw = cf.crop_width * sx
    ch = cf.crop_height * sy

    # 目标整数宽高
    cw_int = int(round(cw))
    ch_int = int(round(ch))

    # 初始左上角 / 右下角
    x1 = int(round(cx - cw / 2.0))
    y1 = int(round(cy - ch / 2.0))
    x2 = x1 + cw_int
    y2 = y1 + ch_int

    # 左越界 -> 往右平移
    if x1 < 0:
        shift = -x1
        x1 = 0
        x2 = x1 + cw_int
    # 上越界 -> 往下平移
    if y1 < 0:
        shift = -y1
        y1 = 0
        y2 = y1 + ch_int

    # 右越界 -> 往左平移
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift
        x2 = frame_w
        if x1 < 0:
            x1 = 0
    # 下越界 -> 往上平移
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift
        y2 = frame_h
        if y1 < 0:
            y1 = 0

    # 最终再 clamp 防御（至少 1 像素宽高）
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))

    return x1, y1, x2, y2


import os
from typing import Callable, List, Optional
import cv2
import numpy as np
def export_stabilized_video(
    input_video_path: str,
    crop_frames: List[CropFrame],
    output_video_path: str,
    work_width: int,
    work_height: int,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> None:
    """
    根据 crop_frames 对输入视频进行逐帧裁切，生成稳定后的视频。

    Args:
        input_video_path: 原始视频文件路径。
        crop_frames: 规划好的裁切轨迹（工作分辨率下的坐标）。
        output_video_path: 输出视频文件路径。
        work_width: 规划时使用的“工作分辨率宽度”（比如 TrackEngine.scale_width）。
        work_height: 规划时使用的“工作分辨率高度”（比如 TrackEngine.scale_height）。
        progress_cb: 可选回调函数，用于更新界面的进度百分比 progress_cb(0~100.0)。
    """

    if not crop_frames:
        raise RuntimeError("export_stabilized_video: crop_frames is empty")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 1e-3:
        fps = 30.0

    total_frames = min(len(crop_frames), total_frames_cap if total_frames_cap > 0 else len(crop_frames))
    if total_frames == 0:
        cap.release()
        raise RuntimeError("export_stabilized_video: no frames to export")

    # 通过工作->原始的缩放估算输出分辨率
    sx = frame_w / float(work_width)
    sy = frame_h / float(work_height)

    first_cf = crop_frames[0]
    out_w = int(round(first_cf.crop_width * sx))
    out_h = int(round(first_cf.crop_height * sy))

    # 不超过原视频分辨率
    out_w = min(out_w, frame_w)
    out_h = min(out_h, frame_h)

    # 对齐到偶数尺寸（编码器友好）
    if out_w % 2 == 1:
        out_w -= 1
    if out_h % 2 == 1:
        out_h -= 1
    if out_w <= 0 or out_h <= 0:
        cap.release()
        raise RuntimeError("export_stabilized_video: invalid output size")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_video_path}")

    frame_idx = 0
    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            cf = crop_frames[frame_idx]

            x1, y1, x2, y2 = _map_crop_to_full_res(
                frame_w=frame_w,
                frame_h=frame_h,
                work_w=work_width,
                work_h=work_height,
                cf=cf,
            )

            crop = frame[y1:y2, x1:x2]

            ch, cw = crop.shape[:2]
            if cw != out_w or ch != out_h:
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(crop)

            frame_idx += 1

            if progress_cb is not None:
                percent = 100.0 * frame_idx / float(total_frames)
                progress_cb(percent)

    finally:
        cap.release()
        writer.release()

