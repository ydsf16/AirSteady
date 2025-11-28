import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from ultralytics import YOLO

from utils import get_airsteady_cache_dir


def resource_path(relative_path: str) -> str:
    """
    兼容普通运行和 PyInstaller 打包后的资源路径.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # PyInstaller 临时目录
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ======================================================================
# 跟踪部分
# ======================================================================

@dataclass
class TrackFrame:
    """
    单帧跟踪结果（在缩放后的工作分辨率下）.
    """
    timestamp: float
    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    valid: bool       # 是否是“唯一目标”的有效观测
    num_obj: int      # 当前帧该类别目标数量


class TrackEngine:
    """
    AirSteady 跟踪引擎：
    - 负责打开原始视频
    - 限制尺寸到工作分辨率 (scale_width, scale_height)
    - 调用 YOLO 跟踪飞机，返回 TrackFrame 和可视化图像
    - 同时把工作分辨率视频写入临时文件 tmp_track.mp4，供后续稳像规划使用
    """

    def __init__(
        self,
        model_path: str = "model.pt",
    ):
        # 解析模型路径（默认 core/model/model.pt）
        if not os.path.isabs(model_path):
            model_path = resource_path(os.path.join("model", model_path))

        self.model_path = model_path
        self.tracker_cfg = resource_path(os.path.join("model", "tracker.yaml"))

        # YOLO 模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 视频相关属性
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.fps: float = 25.0
        self.width: int = 0
        self.height: int = 0
        self.scale_width: int = 0
        self.scale_height: int = 0
        self.scale_ratio: float = 1.0

        self.total_frames: int = 0
        self.current_frame_idx: int = 0

        self.names = self.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

        # 预览时限制最大尺寸
        self.max_width = 600
        self.max_height = 600

        # 跟踪结果（完整轨迹）
        self.track_results: List[TrackFrame] = []

        # 模型 warm-up（避免第一帧巨卡）
        print("Loading model")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model.predict(fake_frame, device=self.device, verbose=False)
        print("Model loaded")

        # 预览视频（缩放后）临时路径
        self.tmp_video_path = os.path.join(get_airsteady_cache_dir(), "tmp_track.mp4")
        self.temp_writer: Optional[cv2.VideoWriter] = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 当前选择的类别
        self.target_class_name: str = "airplane"
        self.target_class_id: Optional[int] = None

    # ------------------------------------------------------------------
    # 打开视频
    # ------------------------------------------------------------------
    def open_video(self, path: str):
        """
        打开视频并初始化 fps/尺寸/缩放比例/VideoWriter.
        """
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

        # 计算缩放比例（保证宽高都不超过 max_width/max_height）
        self.scale_ratio = 1.0
        if self.width > self.max_width:
            self.scale_ratio = min(self.scale_ratio, float(self.max_width) / self.width)
        if self.height > self.max_height:
            self.scale_ratio = min(self.scale_ratio, float(self.max_height) / self.height)

        self.scale_width = int(self.width * self.scale_ratio)
        self.scale_height = int(self.height * self.scale_ratio)

        print("Scale ratio", self.scale_ratio)
        print("scale_width", self.scale_width)
        print("scale_height", self.scale_height)

        # 清空旧的跟踪结果
        self.track_results.clear()

        # 打开临时写入器（写缩放后的视频）
        self.temp_writer = cv2.VideoWriter(
            self.tmp_video_path,
            self.fourcc,
            self.fps,
            (self.scale_width, self.scale_height),
        )

    def finished(self):
        """
        结束当前视频的写入和读取。
        """
        if self.temp_writer is not None:
            self.temp_writer.release()
            self.temp_writer = None
            print("finished track writter")

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print("finished track cap")

    # ------------------------------------------------------------------
    # 单步前向：读取下一帧，做 YOLO + 跟踪
    # ------------------------------------------------------------------
    def next_frame(self, target_class: str = "airplane") -> Tuple[Optional[np.ndarray], Optional[TrackFrame]]:
        """
        读取下一帧并处理（用于在线预览）。
        返回:
            vis_frame: 已绘制角标/十字的 BGR 图像
            track_frame: TrackFrame 或 None
        """
        if self.cap is None:
            return None, None

        success, frame = self.cap.read()
        if not success:
            return None, None

        # 缩放到工作分辨率
        if self.scale_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (self.scale_width, self.scale_height),
                interpolation=cv2.INTER_AREA,
            )

        # 写入缩放后的视频帧到临时文件
        if self.temp_writer is not None:
            self.temp_writer.write(frame)

        # class name -> class id.
        self.target_class_name = target_class
        self.target_class_id = None
        for i, name in self.names.items():
            if name == self.target_class_name:
                self.target_class_id = int(i)
                break

        # 当前帧索引（0-based）
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # YOLO 跟踪（persist=True 保留 track id）
        result = self.model.track(
            frame,
            persist=True,
            verbose=False,
            device=self.device,
            tracker=self.tracker_cfg,
        )[0]

        vis_frame = frame.copy()
        boxes = result.boxes
        selected_xywh: List[Tuple[float, float, float, float]] = []
        selected_confs: List[float] = []

        # ---------- STEP1: 提取指定类别的 bbox + conf ----------
        if boxes is not None and boxes.xywh is not None and self.target_class_id is not None:
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()

            for (cx, cy, w_box, h_box), conf, cls_id in zip(xywh, confs, cls_ids):
                if cls_id == self.target_class_id:
                    selected_xywh.append((float(cx), float(cy), float(w_box), float(h_box)))
                    selected_confs.append(float(conf))

        num_obj = len(selected_xywh)

        # ---------- STEP2: 绘制角标 + 中间十字 ----------
        colors = [
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 0),
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
            corner_len = max(1, int(0.2 * min(w_i, h_i)))
            cross_len = max(1, int(0.15 * min(w_i, h_i)))

            # 四个角
            cv2.line(vis_frame, (x1, y1), (x1 + corner_len, y1), color, line_thickness)
            cv2.line(vis_frame, (x1, y1), (x1, y1 + corner_len), color, line_thickness)

            cv2.line(vis_frame, (x2, y1), (x2 - corner_len, y1), color, line_thickness)
            cv2.line(vis_frame, (x2, y1), (x2, y1 + corner_len), color, line_thickness)

            cv2.line(vis_frame, (x1, y2), (x1 + corner_len, y2), color, line_thickness)
            cv2.line(vis_frame, (x1, y2), (x1, y2 - corner_len), color, line_thickness)

            cv2.line(vis_frame, (x2, y2), (x2 - corner_len, y2), color, line_thickness)
            cv2.line(vis_frame, (x2, y2), (x2, y2 - corner_len), color, line_thickness)

            # 中间十字
            cx_i = int(cx)
            cy_i = int(cy)
            cv2.line(vis_frame, (cx_i - cross_len, cy_i), (cx_i + cross_len, cy_i), color, line_thickness)
            cv2.line(vis_frame, (cx_i, cy_i - cross_len), (cx_i, cy_i + cross_len), color, line_thickness)

        # ---------- STEP3: 组织 TrackFrame ----------
        if hasattr(self, "fps") and self.fps > 1e-6:
            timestamp = float(self.current_frame_idx / self.fps)
        else:
            timestamp = 0.0

        if num_obj == 1:
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
            # 0 个 或 >1 个目标 → 视为无效（保持画面，但不用于后续规划）
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


# ======================================================================
# 裁切轨迹规划部分
# ======================================================================

@dataclass
class CropFrame:
    """
    单帧裁切结果（工作分辨率下的参数）.

    clamp 含义：
    - 仅当该帧是“真实有效观测帧”（effective_mask=True），且在构造裁切框时
      需要把中心点 clamp 到允许范围内时，才会被标记为 True。
    - 用来给 UI 提示“已经碰到当前裁切能力上限了，建议增大裁切保留比例重算”。
    """
    timestamp: float
    frame_idx: int
    crop_center_x: float
    crop_center_y: float
    crop_width: float
    crop_height: float
    scale: float   # 当前工作分辨率缩放因子（外层可用 orig_W / work_W 做映射）
    clamp: bool    # 是否触碰了裁切极限（仅对有效 tracking 帧标记）


def build_observations_and_weights(
    track_result: List[TrackFrame],
    img_width: int,
    img_height: int,
    max_crop_ratio: float,
    density_window_sec: float = 2.0,
    density_min_ratio: float = 0.3,
    long_gap_sec: float = 5.0,
    recenter_weight_scale: float = 0.01,
):
    """
    Step1: 从 TrackFrame 构造观测 + 权重 + 各种 mask.
    包含：
    - 基于 max_crop_ratio 的几何 clamp 可行域
    - 观测密度过滤
    - 长时间完全无观测时的“回中”观测
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

    # ---- Step1: max_crop_ratio 下的几何可行区 clamp ----
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

    # ---- 规则 1~4: 真实有效观测 real_mask ----
    real_mask = valid_raw.copy()
    real_mask &= (num_obj_raw == 1)

    # 边缘贴边 & 过大 bbox 过滤
    margin_left = cx_raw - 0.5 * w_raw
    margin_right = W - (cx_raw + 0.5 * w_raw)
    margin_top = cy_raw - 0.5 * h_raw
    margin_bottom = H - (cy_raw + 0.5 * h_raw)
    min_margin_x = np.minimum(margin_left, margin_right)
    min_margin_y = np.minimum(margin_top, margin_bottom)
    edge_thresh_x = 0.05 * W
    edge_thresh_y = 0.05 * H
    bbox_at_edge = (min_margin_x < edge_thresh_x) | (min_margin_y < edge_thresh_y)
    bbox_too_large = (w_raw > 0.95 * W) | (h_raw > 0.95 * H)
    bad_bbox = bbox_at_edge | bbox_too_large
    real_mask &= ~bad_bbox

    # 基础权重（后面还会叠加密度/Huber）
    conf_norm = np.clip(conf_raw, 0.0, 1.0)
    base_weights = np.zeros(n, dtype=float)
    base_weights[real_mask] = conf_norm[real_mask]

    # -------- A) 观测密度筛选 --------
    if n > 0 and density_window_sec > 0.0 and density_min_ratio > 0.0:
        if timestamps[-1] > timestamps[0]:
            fps_est = (n - 1) / (timestamps[-1] - timestamps[0])
        else:
            fps_est = 30.0

        window_frames = max(1, int(density_window_sec * fps_est))
        kernel = np.ones(window_frames, dtype=float)
        density = np.convolve(real_mask.astype(float), kernel, mode="same")

        density_min_ratio = float(np.clip(density_min_ratio, 0.0, 1.0))
        min_count = max(1, int(round(density_min_ratio * window_frames)))

        low_density = density < float(min_count)
        base_weights[low_density] = 0.0
        real_mask[low_density] = False

    # -------- B) 长时间完全无观测：回中处理 --------
    if n > 0 and long_gap_sec > 0.0 and recenter_weight_scale > 0.0:
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
            start = i
            while i < n and not real_mask[i]:
                i += 1
            end = i

            dur = timestamps[end - 1] - timestamps[start]
            if dur >= long_gap_sec:
                obs_cx[start:end] = W / 2.0
                obs_cy[start:end] = H / 2.0
                base_weights[start:end] = recenter_w
                # real_mask 不改，表示这些是“回中辅助观测”，不计入有效 tracking

    weights = base_weights.copy()
    solver_effective_mask = weights > 0

    # 如果所有权重都是 0，退化成简单平滑（避免后面全 0）
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
        "weights": weights,
        "real_effective_mask": real_mask,
        "solver_effective_mask": solver_effective_mask,
        "W": W,
        "H": H,
    }


# ----------------------------------------------------------------------
# 稀疏平滑求解（带 Huber 的 IRLS）
# ----------------------------------------------------------------------
def _solve_axis_sparse_linear(
    obs: np.ndarray,
    weights: np.ndarray,
    smooth_factor: float,
) -> np.ndarray:
    """
    只做纯 L2 最小二乘的单轴平滑（无鲁棒核）.
    min  Σ w_i (p_i - obs_i)^2
       + λ_pos Σ (p_i - p_{i-1})^2
       + λ_vel Σ (p_i - 2p_{i-1} + p_{i-2})^2
    """
    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return obs.copy()

    smooth_factor = float(np.clip(smooth_factor, 0.0, 1.0))
    lambda_pos = 10.0 * (smooth_factor ** 2)
    lambda_vel = 5.0 * (smooth_factor ** 2)

    diag0 = np.zeros(n, dtype=float)
    diag1 = np.zeros(n - 1, dtype=float)
    diag2 = np.zeros(n - 2, dtype=float)
    b = np.zeros(n, dtype=float)

    # 数据项
    diag0 += weights
    b += weights * obs

    # 位置平滑项
    if lambda_pos > 0.0:
        for i in range(1, n):
            diag0[i] += lambda_pos
            diag0[i - 1] += lambda_pos
            diag1[i - 1] += -lambda_pos

    # 速度平滑项
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
    - base_weights: 原始置信度/权重
    - huber_delta: 残差阈值（像素），大于该值的残差权重会衰减
    """
    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return obs.copy()

    base_w = np.clip(base_weights, 0.0, np.inf)

    # 初始解：纯 L2 解
    w_cur = base_w.copy()
    p = _solve_axis_sparse_linear(obs, w_cur, smooth_factor)

    for _ in range(max_iter):
        r = p - obs
        abs_r = np.abs(r)

        robust_factor = np.ones_like(abs_r, dtype=float)
        mask = abs_r > 1e-6
        big = mask & (abs_r > huber_delta)
        robust_factor[big] = huber_delta / abs_r[big]

        w_cur = base_w * robust_factor
        if np.all(w_cur == 0):
            w_cur[:] = 1.0

        p = _solve_axis_sparse_linear(obs, w_cur, smooth_factor)

    return p


def solve_smooth_trajectory(
    obs_cx: np.ndarray,
    obs_cy: np.ndarray,
    weights: np.ndarray,
    smooth_factor: float,
    huber_delta: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Huber-IRLS 版本的双轴平滑。
    - smooth_factor 非常小时，直接返回 obs（避免矩阵奇异）
    """
    n = len(obs_cx)
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    smooth_factor = float(smooth_factor)
    if smooth_factor <= 1e-6:
        # 这里 obs_cx/obs_cy 已经经过 clamp + 回中，不会是 NaN
        return obs_cx.astype(float).copy(), obs_cy.astype(float).copy()

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
    Step2: 只用有效帧(effective_mask==True)来计算全局裁切比例.
    返回 [0, max_crop_ratio] 内的全局 crop_ratio.
    """
    max_crop_ratio = float(np.clip(max_crop_ratio, 0.0, 0.9))
    if max_crop_ratio <= 1e-6:
        return 0.0

    if np.any(effective_mask):
        px_eff = px[effective_mask]
        py_eff = py[effective_mask]
    else:
        px_eff = px
        py_eff = py

    valid = np.isfinite(px_eff) & np.isfinite(py_eff)
    if not np.any(valid):
        # 全是 NaN，退化成不裁切
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
    Step4: 用平滑后的 px, py + 全局 crop_ratio 生成 list[CropFrame].
    - 所有帧都会输出 CropFrame
    - clamp 标记只对 effective_mask==True 的帧有效
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

        # 所有帧都要保证裁切框不出界
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
                scale=1.0,
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
    Debug 可视化：中心轨迹 + 裁切边界.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[planning debug] matplotlib not available: {e}")
        return

    t = timestamps
    eff = effective_mask
    crop_ratio = float(np.clip(crop_ratio, 0.0, 1.0))

    crop_width = W * (1.0 - crop_ratio)
    crop_height = H * (1.0 - crop_ratio)
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0

    left_x = np.clip(px - half_w, 0.0, W)
    right_x = np.clip(px + half_w, 0.0, W)
    top_y = np.clip(py - half_h, 0.0, H)
    bottom_y = np.clip(py + half_h, 0.0, H)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # X
    ax = axes[0]
    ax.set_title("Crop Planning Debug - X")
    ax.plot(t, cx_raw, "k.", alpha=0.3, label="raw cx")
    ax.plot(t, obs_cx, "b-", alpha=0.6, label="obs cx (clamped)")
    ax.plot(t[eff], obs_cx[eff], "bo", alpha=0.9, label="effective obs")
    ax.plot(t, px, "r-", alpha=0.9, label="smoothed cx")
    ax.plot(t, left_x, "--", color="gray", alpha=0.5, label="crop left")
    ax.plot(t, right_x, "--", color="gray", alpha=0.5, label="crop right")
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.axhline(W, color="black", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("cx (px)")
    ax.set_ylim(0.0, W)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Y
    ay = axes[1]
    ay.set_title("Crop Planning Debug - Y")
    ay.plot(t, cy_raw, "k.", alpha=0.3, label="raw cy")
    ay.plot(t, obs_cy, "b-", alpha=0.6, label="obs cy (clamped)")
    ay.plot(t[eff], obs_cy[eff], "bo", alpha=0.9, label="effective obs")
    ay.plot(t, py, "r-", alpha=0.9, label="smoothed cy")
    ay.plot(t, top_y, "--", color="gray", alpha=0.5, label="crop top")
    ay.plot(t, bottom_y, "--", color="gray", alpha=0.5, label="crop bottom")
    ay.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ay.axhline(H, color="black", linewidth=0.5, alpha=0.5)
    ay.set_ylabel("cy (px)")
    ay.set_xlabel("time (s)")
    ay.set_ylim(0.0, H)
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
    规划整段视频的裁切轨迹（工作分辨率下）.
    """
    if not track_result:
        return []

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
    real_effective_mask = ctx["real_effective_mask"]
    W = ctx["W"]
    H = ctx["H"]

    # Step3: Huber + 五对角平滑
    px, py = solve_smooth_trajectory(
        obs_cx=obs_cx,
        obs_cy=obs_cy,
        weights=weights,
        smooth_factor=smooth_factor,
        huber_delta=10.0,
    )

    # Step2: 计算全局裁切比例（只用真实有效观测）
    crop_ratio = compute_global_crop_ratio(
        px=px,
        py=py,
        W=W,
        H=H,
        max_crop_ratio=max_crop_ratio,
        effective_mask=real_effective_mask,
    )

    # Step4: 构造 per-frame CropFrame
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


# ======================================================================
# 导出裁切后视频
# ======================================================================

def _map_crop_to_full_res(
    frame_w: int,
    frame_h: int,
    work_w: int,
    work_h: int,
    cf: CropFrame,
) -> Tuple[int, int, int, int]:
    """
    将 CropFrame 中的中心/宽高，从工作分辨率 (work_w, work_h)
    映射到原始帧分辨率 (frame_w, frame_h)，并做边界 clamp.
    """
    sx = frame_w / float(work_w)
    sy = frame_h / float(work_h)

    cx = cf.crop_center_x * sx
    cy = cf.crop_center_y * sy
    cw = cf.crop_width * sx
    ch = cf.crop_height * sy

    cw_int = int(round(cw))
    ch_int = int(round(ch))

    x1 = int(round(cx - cw / 2.0))
    y1 = int(round(cy - ch / 2.0))
    x2 = x1 + cw_int
    y2 = y1 + ch_int

    if x1 < 0:
        shift = -x1
        x1 = 0
        x2 = x1 + cw_int
    if y1 < 0:
        shift = -y1
        y1 = 0
        y2 = y1 + ch_int

    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift
        x2 = frame_w
        if x1 < 0:
            x1 = 0
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift
        y2 = frame_h
        if y1 < 0:
            y1 = 0

    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))

    return x1, y1, x2, y2


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

    # 估算输出分辨率（基于第一帧裁切宽高）
    sx = frame_w / float(work_width)
    sy = frame_h / float(work_height)

    first_cf = crop_frames[0]
    out_w = int(round(first_cf.crop_width * sx))
    out_h = int(round(first_cf.crop_height * sy))

    out_w = min(out_w, frame_w)
    out_h = min(out_h, frame_h)

    if out_w % 2 == 1:
        out_w -= 1
    if out_h % 2 == 1:
        out_h -= 1

    if out_w <= 0 or out_h <= 0 or not np.isfinite(out_w) or not np.isfinite(out_h):
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
