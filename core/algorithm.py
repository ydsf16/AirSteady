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
from asset_loader import get_default_assets_path, load_model_and_config


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

    # 新增：光流估计的帧间位移（当前帧相对上一帧）
    dx: float = float("nan")
    dy: float = float("nan")
    has_delta: bool = False   # True 表示 dx, dy 有效


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
        # if not os.path.isabs(model_path):
        #     model_path = resource_path(os.path.join("model", model_path))
        # self.model_path = model_path
        # self.tracker_cfg = resource_path(os.path.join("model", "tracker.yaml"))
        # # YOLO 模型
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = YOLO(self.model_path)
        # self.model.to(self.device)

        # ===== 1. 从加密包中解密模型 & 配置，并加载 YOLO =====
        enc_path = get_default_assets_path()
        model, tracker_cfg_path, device = load_model_and_config(enc_path)

        self.model = YOLO("yolo11s-seg.pt")
        self.device = device
        self.tracker_cfg = tracker_cfg_path  # 这里就是 yaml 文件路径

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
        self.max_width = 2000
        self.max_height = 2000

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

        # ---------- 光流跟踪状态 ----------
        self.of_prev_gray: Optional[np.ndarray] = None
        self.of_prev_pts: Optional[np.ndarray] = None  # 形状 (N, 1, 2)
        self.of_tracking: bool = False

        # 光流 RANSAC 参数（纯平移模型）
        self.of_min_track_points: int = 50
        self.of_ransac_threshold: float = 2.0
        self.of_ransac_max_iters: int = 300
        self.of_min_inlier_ratio: float = 0.2
    
    
    def estimate_translation_ransac(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        max_iters: int = 300,
        thresh: float = 2.0,
    ) -> Tuple[bool, float, float, float, np.ndarray]:
        """
        估计二维纯平移：p2 = p1 + t，通过 RANSAC 拟合 t = (dx, dy).

        Args:
            pts1, pts2: 形状 (N, 2)，对应光流匹配点.
        Returns:
            ok: 是否成功
            dx, dy: 平移
            inlier_ratio: 内点比例
            inlier_mask: bool 数组，长度 N
        """
        assert pts1.shape == pts2.shape
        N = pts1.shape[0]
        if N < 5:
            return False, 0.0, 0.0, 0.0, np.zeros(N, dtype=bool)

        diffs = pts2 - pts1  # (N, 2)
        best_inliers = np.zeros(N, dtype=bool)
        best_count = 0
        best_dx, best_dy = 0.0, 0.0

        for _ in range(max_iters):
            idx = np.random.randint(0, N)
            dx0, dy0 = diffs[idx]
            residuals = np.linalg.norm(diffs - np.array([dx0, dy0]), axis=1)
            inliers = residuals < thresh
            cnt = int(inliers.sum())
            if cnt > best_count:
                best_count = cnt
                best_inliers = inliers
                best_dx, best_dy = dx0, dy0

        if best_count == 0:
            return False, 0.0, 0.0, 0.0, best_inliers

        # 对内点再做一次均值估计
        mean_dx, mean_dy = diffs[best_inliers].mean(axis=0)
        inlier_ratio = float(best_count) / float(N)
        return True, float(mean_dx), float(mean_dy), inlier_ratio, best_inliers


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
        读取下一帧并处理（用于在线预览）.

        返回:
            vis_frame: 已绘制角标/十字的 BGR 图像
            track_frame: TrackFrame 或 None
        """
        if self.cap is None:
            return None, None

        success, frame = self.cap.read()
        if not success or frame is None:
            return None, None

        # 缩放到工作分辨率
        if self.scale_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (self.scale_width, self.scale_height),
                interpolation=cv2.INTER_AREA,
            )

        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        # 在右下角打黑块，遮住角标（写入 temp 视频使用）
        logo_w_ratio = 0.10
        logo_h_ratio = 0.18
        margin_ratio = 0.02

        logo_w = int(w * logo_w_ratio)
        logo_h = int(h * logo_h_ratio)
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x2 = w - margin_x
        x1 = max(0, x2 - logo_w)
        y2 = h - margin_y
        y1 = max(0, y2 - logo_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

        # 灰度图（光流用）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 当前帧索引（0-based）
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # --------- STEP0: 使用上一帧状态做光流，估计 delta ----------
        delta_x = float("nan")
        delta_y = float("nan")
        has_delta = False

        if self.of_tracking and self.of_prev_gray is not None and self.of_prev_pts is not None:
            # Lucas-Kanade 金字塔光流
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.of_prev_gray,
                gray,
                self.of_prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_pts is not None and status is not None:
                status = status.reshape(-1)
                prev_pts_flat = self.of_prev_pts.reshape(-1, 2)
                next_pts_flat = next_pts.reshape(-1, 2)

                good_mask = (status == 1)
                pts1 = prev_pts_flat[good_mask]
                pts2 = next_pts_flat[good_mask]

                if pts1.shape[0] >= self.of_min_track_points:
                    ok, dx, dy, inlier_ratio, inlier_mask = self.estimate_translation_ransac(
                        pts1, pts2,
                        max_iters=self.of_ransac_max_iters,
                        thresh=self.of_ransac_threshold,
                    )
                    if ok and inlier_ratio >= self.of_min_inlier_ratio:
                        delta_x = dx
                        delta_y = dy
                        has_delta = True

                        # 只保留内点，更新状态
                        pts2_inlier = pts2[inlier_mask]
                        self.of_prev_pts = pts2_inlier.reshape(-1, 1, 2)
                        self.of_prev_gray = gray.copy()
                    else:
                        # RANSAC 判定失败，停止光流
                        self.of_tracking = False
                        self.of_prev_gray = None
                        self.of_prev_pts = None
                else:
                    # 可用点太少
                    self.of_tracking = False
                    self.of_prev_gray = None
                    self.of_prev_pts = None
            else:
                self.of_tracking = False
                self.of_prev_gray = None
                self.of_prev_pts = None

        # --------- STEP1: YOLO 检测（每帧都跑） ----------
        self.target_class_name = target_class
        self.target_class_id = None
        for i, name in self.names.items():
            if name == self.target_class_name:
                self.target_class_id = int(i)
                break

        result = self.model.track(
            frame,
            persist=True,
            verbose=False,
            device=self.device,
            tracker=self.tracker_cfg,
            imgsz=960,
        )[0]

        # 写入缩放后的视频帧到临时文件（可视化时用的是 vis_frame，而不是打码后的 frame）
        if self.temp_writer is not None:
            self.temp_writer.write(vis_frame)

        boxes = result.boxes
        masks = getattr(result, "masks", None)  # YOLOv11-seg 时可能存在
        selected_xywh: List[Tuple[float, float, float, float]] = []
        selected_confs: List[float] = []
        selected_masks: List[Optional[np.ndarray]] = []

        if boxes is not None and boxes.xywh is not None and self.target_class_id is not None:
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.int().cpu().numpy()

            # seg mask 数据（如果有）
            mask_data = None
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_data = masks.data.cpu().numpy()  # (N, Hm, Wm)
                print("has mask")

            for idx, ((cx, cy, w_box, h_box), conf, cls_id) in enumerate(zip(xywh, confs, cls_ids)):
                if cls_id == self.target_class_id:
                    selected_xywh.append((float(cx), float(cy), float(w_box), float(h_box)))
                    selected_confs.append(float(conf))

                    if mask_data is not None and idx < mask_data.shape[0]:
                        m = mask_data[idx]
                        # 转成 uint8 mask，必要时 resize 到当前帧尺寸
                        if m.shape != gray.shape:
                            m_resized = cv2.resize(
                                m.astype(np.float32),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            m_bin = (m_resized > 0.5).astype(np.uint8)
                        else:
                            m_bin = (m > 0.5).astype(np.uint8)
                        selected_masks.append(m_bin)
                    else:
                        selected_masks.append(None)

        num_obj = len(selected_xywh)

        # --------- STEP2: 绘制角标 + 中间十字 ----------
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

        # --------- STEP3: 根据检测情况重置光流参考帧（只在 num_obj == 1 时） ----------
        if num_obj == 1:
            cx, cy, w_box, h_box = selected_xywh[0]

            # 选一个 mask：优先用语义 mask，退化到 bbox
            mask_for_of = np.zeros_like(gray, dtype=np.uint8)
            if selected_masks and selected_masks[0] is not None:
                mask_for_of = (selected_masks[0] > 0).astype(np.uint8) * 255
            else:
                x1 = int(max(0, cx - w_box / 2.0))
                y1 = int(max(0, cy - h_box / 2.0))
                x2 = int(min(w - 1, cx + w_box / 2.0))
                y2 = int(min(h - 1, cy + h_box / 2.0))
                mask_for_of[y1:y2, x1:x2] = 255

            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=2000,
                qualityLevel=0.01,
                minDistance=1.0,
                mask=mask_for_of,
            )

            if pts is not None and pts.shape[0] >= self.of_min_track_points:
                self.of_prev_gray = gray.copy()
                self.of_prev_pts = pts.reshape(-1, 1, 2)
                self.of_tracking = True
            else:
                self.of_prev_gray = None
                self.of_prev_pts = None
                self.of_tracking = False

        # --------- STEP4: 组织 TrackFrame（带 delta） ----------
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
                dx=delta_x,
                dy=delta_y,
                has_delta=bool(has_delta),
            )
        else:
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
                dx=delta_x,
                dy=delta_y,
                has_delta=bool(has_delta),
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
    smooth_factor: float = 0.5,  # 保留参数占位
    debug: bool = False,
) -> List[CropFrame]:
    """
    简化版裁切规划（delta 积分版）.

    逻辑：
    1) 默认所有帧的裁切中心 = 图像正中间。
    2) 对于 has_delta 连续的段落：
       - 以该段的第一帧为基准帧：
         center[s] = 基准帧的检测中心（若无则用图像中心）
         后续帧中心 = 前一帧中心 + delta（逐帧累加）
    3) 不在任何 delta 连续段内的帧，一律保持图像中心。
    4) 裁切宽高由 max_crop_ratio 决定（不再做全局优化）。
    """
    if not track_result:
        return []

    n = len(track_result)
    W = float(img_width)
    H = float(img_height)

    timestamps = np.array([f.timestamp for f in track_result], dtype=float)
    frame_idx_arr = np.array([f.frame_idx for f in track_result], dtype=int)
    cx_det = np.array([f.cx for f in track_result], dtype=float)
    cy_det = np.array([f.cy for f in track_result], dtype=float)
    valid_det = np.array([f.valid for f in track_result], dtype=bool)
    dx_arr = np.array([f.dx for f in track_result], dtype=float)
    dy_arr = np.array([f.dy for f in track_result], dtype=float)
    has_delta_arr = np.array([f.has_delta for f in track_result], dtype=bool)

    # 1) 默认所有帧中心 = 图像中心
    center_x = np.full(n, W / 2.0, dtype=float)
    center_y = np.full(n, H / 2.0, dtype=float)

    # 2) 找连续 has_delta 段落，做积分
    # 注意：has_delta[i] 表示 i 与 i-1 之间有 delta
    i = 1
    while i < n:
        # 段落开始条件：当前帧有 delta，且前一帧没有（或 i == 1）
        if not has_delta_arr[i] or (i > 1 and has_delta_arr[i - 1]):
            i += 1
            continue

        base = i - 1  # 段落的第一帧（无 delta，作为 ref）
        # 基准中心：优先用该帧的检测中心
        if valid_det[base]:
            ref_cx = cx_det[base]
            ref_cy = cy_det[base]
        else:
            ref_cx = W / 2.0
            ref_cy = H / 2.0

        center_x[base] = ref_cx
        center_y[base] = ref_cy

        j = i
        prev_cx = ref_cx
        prev_cy = ref_cy

        while j < n and has_delta_arr[j]:
            dx = dx_arr[j]
            dy = dy_arr[j]
            if not np.isfinite(dx) or not np.isfinite(dy):
                break
            prev_cx = prev_cx + dx
            prev_cy = prev_cy + dy
            center_x[j] = prev_cx
            center_y[j] = prev_cy
            j += 1

        # 你如果希望“短段落”直接丢弃，可以在这里判断段长 j-base
        # 目前不做最小长度限制，完全按 delta 连续性来。

        i = j + 1

    # 3) 根据 max_crop_ratio 决定裁切宽高（简单映射）
    crop_ratio = float(np.clip(max_crop_ratio, 0.0, 0.9))
    crop_width = W * (1.0 - crop_ratio)
    crop_height = H * (1.0 - crop_ratio)

    crop_frames: List[CropFrame] = []
    for k in range(n):
        crop_frames.append(
            CropFrame(
                timestamp=float(timestamps[k]),
                frame_idx=int(frame_idx_arr[k]),
                crop_center_x=float(center_x[k]),
                crop_center_y=float(center_y[k]),
                crop_width=float(crop_width),
                crop_height=float(crop_height),
                scale=1.0,
                clamp=False,   # 不再做 clamp，黑边交给导出函数处理
            )
        )

    # debug 可视化简单画一下 center_x/center_y
    if debug:
        try:
            import matplotlib.pyplot as plt
            t = timestamps
            plt.figure(figsize=(10, 4))
            plt.subplot(2, 1, 1)
            plt.title("Center X with delta integration")
            plt.plot(t, center_x, "r-")
            plt.axhline(W / 2.0, color="gray", linestyle="--")
            plt.subplot(2, 1, 2)
            plt.title("Center Y with delta integration")
            plt.plot(t, center_y, "b-")
            plt.axhline(H / 2.0, color="gray", linestyle="--")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[planning debug] matplotlib not available: {e}")

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

    # 基本尺寸检查，防止除 0 或负数
    if frame_w <= 0 or frame_h <= 0 or work_w <= 0 or work_h <= 0:
        raise RuntimeError(
            f"_map_crop_to_full_res: invalid size "
            f"frame=({frame_w}x{frame_h}), work=({work_w}x{work_h})"
        )

    # 映射比例（一般 preview 时 frame == work，这里只是做保护）
    sx = frame_w / float(work_w)
    sy = frame_h / float(work_h)

    cx = cf.crop_center_x * sx
    cy = cf.crop_center_y * sy
    cw = cf.crop_width * sx
    ch = cf.crop_height * sy

    # 宽高至少 1 像素
    cw_int = max(1, int(round(cw)))
    ch_int = max(1, int(round(ch)))

    # 初始框（可能会出界）
    x1 = int(round(cx - cw / 2.0))
    y1 = int(round(cy - ch / 2.0))
    x2 = x1 + cw_int
    y2 = y1 + ch_int

    # 左上越界 → 往右/下平移
    if x1 < 0:
        shift = -x1
        x1 = 0
        x2 = x1 + cw_int
    if y1 < 0:
        shift = -y1
        y1 = 0
        y2 = y1 + ch_int

    # 右下越界 → 往左/上平移
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

    # 最终 clamp 一下，并保证 x2 > x1, y2 > y1
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))

    # 再做一层 sanity check，防御性编程
    if not (0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h):
        raise RuntimeError(
            f"_map_crop_to_full_res: ROI invalid after clamp: "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
            f"frame=({frame_w}x{frame_h}), work=({work_w}x{work_h}), "
            f"cf_center=({cf.crop_center_x},{cf.crop_center_y}), "
            f"cf_size=({cf.crop_width}x{cf.crop_height})"
        )

    return x1, y1, x2, y2


def export_stabilized_video(
    input_video_path: str,
    crop_frames: List[CropFrame],
    output_video_path: str,
    work_width: int,
    work_height: int,
    progress_cb: Optional[Callable[[float], None]] = None,
    add_brand_watermark: bool = False,  # ⭐ 新增：是否打“AirSteady”水印
) -> None:
    """
    根据 crop_frames 对输入视频进行逐帧裁切，生成稳定后的视频。
    增强版：增加了一堆防御性检查，避免 OpenCV 抛出“Unknown C++ exception”。

    Args:
        ...
        add_brand_watermark: 若为 True，则在每一帧右下角打上
            “稳定处理 · AirSteady「航迹稳拍」” 水印。
    """
    if not crop_frames:
        raise RuntimeError("export_stabilized_video: crop_frames is empty")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"export_stabilized_video: failed to open input video: {input_video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError(
            f"export_stabilized_video: input video has invalid size "
            f"{frame_w}x{frame_h} (path={input_video_path})"
        )

    if fps <= 1e-3 or not np.isfinite(fps):
        fps = 30.0

    # 只导出两者中的最小值，防止 plan 比实际帧数多
    total_frames = min(len(crop_frames), total_frames_cap if total_frames_cap > 0 else len(crop_frames))
    if total_frames == 0:
        cap.release()
        raise RuntimeError("export_stabilized_video: no frames to export (total_frames == 0)")

    # 估算输出分辨率（基于第一帧裁切宽高）
    sx = frame_w / float(work_width) if work_width > 0 else 1.0
    sy = frame_h / float(work_height) if work_height > 0 else 1.0

    first_cf = crop_frames[0]
    out_w = int(round(first_cf.crop_width * sx))
    out_h = int(round(first_cf.crop_height * sy))

    out_w = min(out_w, frame_w)
    out_h = min(out_h, frame_h)

    # 偶数尺寸 + 正数检查
    if out_w % 2 == 1:
        out_w -= 1
    if out_h % 2 == 1:
        out_h -= 1

    if out_w <= 0 or out_h <= 0 or not np.isfinite(out_w) or not np.isfinite(out_h):
        cap.release()
        raise RuntimeError(
            f"export_stabilized_video: invalid output size "
            f"{out_w}x{out_h}, frame={frame_w}x{frame_h}, "
            f"work=({work_width}x{work_height}), "
            f"first_crop=({first_cf.crop_width}x{first_cf.crop_height})"
        )

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"export_stabilized_video: failed to open VideoWriter: {output_video_path}")

    # ⭐ 水印文字基础配置（后面按分辨率自适应缩放）
    brand_text = "Stabilized by AirSteady"
    font = cv2.FONT_HERSHEY_DUPLEX  # 比 SIMPLEX 好看一点
    base_font_scale = 0.7          # 针对 1080p 设计的基准字号
    thickness = 1

    # 这里用 out_w/out_h 来预估位置，真正绘制时会拿当前帧尺寸再安全 clamp 一下
    # text_size, baseline = cv2.getTextSize(brand_text, font, font_scale, thickness)
    # text_w, text_h = text_size

    frame_idx = 0
    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 提前结束，也算正常退出
                break

            cf = crop_frames[frame_idx]

            # ========== 新版：根据中心做平移 + 黑边 ==========
            # work 坐标 -> 原图坐标
            sx = frame_w / float(work_width) if work_width > 0 else 1.0
            sy = frame_h / float(work_height) if work_height > 0 else 1.0

            cx_full = float(cf.crop_center_x) * sx
            cy_full = float(cf.crop_center_y) * sy

            # 目标输出帧中心
            dst_cx = out_w / 2.0
            dst_cy = out_h / 2.0

            dx = dst_cx - cx_full
            dy = dst_cy - cy_full

            M = np.float32([[1.0, 0.0, dx],
                            [0.0, 1.0, dy]])

            # 平移 + 黑边填充
            try:
                crop = cv2.warpAffine(
                    frame,
                    M,
                    (out_w, out_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
            except cv2.error as e:
                raise RuntimeError(
                    f"export_stabilized_video: cv2.warpAffine failed at frame {frame_idx}: {e} | "
                    f"center=({cx_full},{cy_full}), out=({out_w}x{out_h}), "
                    f"frame=({frame_w}x{frame_h}), work=({work_width}x{work_height})"
                ) from e

            # ========= 打水印逻辑保持不变 =========
            if add_brand_watermark:
                h2, w2 = crop.shape[:2]

                base = min(w2, h2)
                scale = base / 1080.0
                font_scale = max(0.5, min(1.8, base_font_scale * scale))

                (text_w, text_h), baseline = cv2.getTextSize(
                    brand_text, font, font_scale, thickness
                )

                margin = int(max(10, base * 0.02))
                x_txt = max(margin, w2 - text_w - margin)
                y_txt = margin + text_h

                overlay = crop.copy()
                pad_x = int(max(6, base * 0.01))
                pad_y = int(max(4, base * 0.007))

                bg_tl = (x_txt - pad_x, y_txt - text_h - pad_y)
                bg_br = (x_txt + text_w + pad_x, y_txt + pad_y)
                cv2.rectangle(overlay, bg_tl, bg_br, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, crop, 0.6, 0, crop)

                shadow_color = (0, 0, 0)
                text_color = (255, 255, 255)

                cv2.putText(
                    crop,
                    brand_text,
                    (x_txt + 1, y_txt + 1),
                    font,
                    font_scale,
                    shadow_color,
                    thickness + 1,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    crop,
                    brand_text,
                    (x_txt, y_txt),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            writer.write(crop)

            frame_idx += 1

            if progress_cb is not None:
                progress_cb(100.0 * frame_idx / float(max(1, total_frames)))

    finally:
        cap.release()
        writer.release()


