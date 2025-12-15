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

# ---------------------------
# 1) Sony S-Log3 -> Scene Linear (reflection)
#    Source: Sony "Technical Summary for S-Gamut3.Cine/S-Log3..." Appendix
# ---------------------------
def slog3_to_linear(x_norm: np.ndarray) -> np.ndarray:
    """
    x_norm: normalized code value in [0,1] (full range), per channel.
    return: scene-linear reflection (can exceed 1.0 in highlights).
    """
    x = x_norm.astype(np.float32)
    t = 171.2102946929 / 1023.0

    hi = (10.0 ** ((x * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01
    lo = (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)

    y = np.where(x >= t, hi, lo)
    return y

# ---------------------------
# 2) Linear -> Rec.709 OETF (display gamma)
# ---------------------------
def linear_to_rec709_oetf(lin: np.ndarray) -> np.ndarray:
    """
    lin: scene-linear (>=0). We'll clamp negative.
    return: Rec709 display-encoded in [0,1] (before quantization).
    """
    L = np.maximum(lin, 0.0).astype(np.float32)
    # Rec.709 OETF
    a = 0.018
    V = np.where(L < a, 4.5 * L, 1.099 * (L ** 0.45) - 0.099)
    return V

# ---------------------------
# 3) Build LUT for uint8 (fast)
# ---------------------------
def build_slog3_to_rec709_lut_u8() -> np.ndarray:
    """
    Returns LUT of shape (256,), mapping uint8 code -> uint8 rec709.
    Assumes full-range mapping 0..255 -> 0..1.
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    lin = slog3_to_linear(x)
    # Optional highlight compression for YOLO friendliness (avoid clipping):
    # You can try one of these:
    # lin = lin / (1.0 + lin)           # Reinhard tone-map
    # lin = np.minimum(lin, 1.0)        # hard clip
    lin = lin / (1.0 + lin)            # recommended default for harsh highlights

    v = linear_to_rec709_oetf(lin)
    v = np.clip(v, 0.0, 1.0)
    lut = np.round(v * 255.0).astype(np.uint8)
    return lut

_SLOG3_TO_709_LUT_U8 = build_slog3_to_rec709_lut_u8()

def slog3_bgr_u8_to_rec709_bgr_u8(bgr_u8: np.ndarray) -> np.ndarray:
    """
    OpenCV BGR uint8 in, BGR uint8 out.
    """
    # Apply per-channel LUT (fast)
    b, g, r = cv2.split(bgr_u8)
    b2 = cv2.LUT(b, _SLOG3_TO_709_LUT_U8)
    g2 = cv2.LUT(g, _SLOG3_TO_709_LUT_U8)
    r2 = cv2.LUT(r, _SLOG3_TO_709_LUT_U8)
    return cv2.merge([b2, g2, r2])

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
        self.max_width  = 2160
        self.max_height = 2160

        # 跟踪结果（完整轨迹）
        self.track_results: List[TrackFrame] = []

        # 模型 warm-up（避免第一帧巨卡）
        print("Loading model")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model.predict(fake_frame, device=self.device, verbose=False)
        print("Model loaded")

        # 预览视频（缩放后）临时路径
        self.tmp_video_path = os.path.join(get_airsteady_cache_dir(), "tmp_track.mp4")
        self.tmp_video_show_path = os.path.join(get_airsteady_cache_dir(), "tmp_track_show.mp4")
        self.temp_writer: Optional[cv2.VideoWriter] = None
        self.temp_writer_show: Optional[cv2.VideoWriter] = None
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # 当前选择的类别
        self.target_class_name: str = "airplane"
        self.target_class_id: Optional[int] = None

        # ---------- 光流跟踪状态 ----------
        self.of_prev_gray: Optional[np.ndarray] = None
        self.of_prev_pts: Optional[np.ndarray] = None  # 形状 (N, 1, 2)
        self.of_tracking: bool = False

        # 光流 RANSAC 参数（纯平移模型）
        self.of_min_track_points: int = 15
        self.of_ransac_threshold: float = 10.0
        self.of_ransac_max_iters: int = 300
        self.of_min_inlier_ratio: float = 0.00000001

        self.debug_mode = True
        self.det_interval = 1
    
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
        num_inliers = best_count
        return True, float(mean_dx), float(mean_dy), inlier_ratio, best_inliers, num_inliers


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
        self.reset_optical_flow_state()

        # 打开临时写入器（写缩放后的视频）
        self.temp_writer = cv2.VideoWriter(
            self.tmp_video_path,
            self.fourcc,
            self.fps,
            (self.scale_width, self.scale_height),
        )

        # Close
        # self.temp_writer_show = cv2.VideoWriter(
        #     self.tmp_video_show_path,
        #     self.fourcc,
        #     self.fps,
        #     (self.scale_width, self.scale_height),
        # )

    def finished(self):
        """
        结束当前视频的写入和读取。
        """
        if self.temp_writer is not None:
            self.temp_writer.release()
            self.temp_writer = None
            print("finished track writter")
        
        if self.temp_writer_show is not None:
            self.temp_writer_show.release()
            self.temp_writer_show = None
            print("finished track writter show")

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print("finished track cap")

        self.reset_optical_flow_state()

    def enhance_for_analysis(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)

        lab2 = cv2.merge([l2, a, b])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # gamma
        gamma = 1.2
        lut = (np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255.0).astype(np.uint8)
        out = cv2.LUT(out, lut)

        # optional denoise (very light)
        # out = cv2.bilateralFilter(out, d=5, sigmaColor=30, sigmaSpace=30)

        return out
    # ------------------------------------------------------------------
    # 单步前向：读取下一帧，做 YOLO + 跟踪
    # ------------------------------------------------------------------
    def next_frame(self, target_class: str = "airplane"):
        """
        Read and process next frame for online preview.

        Returns:
            vis_frame: BGR image with HUD overlay
            track_frame: TrackFrame or None
        """
        import time

        if self.cap is None:
            return None, None

        # ---------------- Profiling helpers ----------------
        profile = getattr(self, "profile_mode", True)
        profile_every = int(getattr(self, "profile_every", 30))  # print every N frames
        t_begin = time.perf_counter()

        def _ms(t0, t1):
            return (t1 - t0) * 1000.0

        # Helper: mask should be mostly inside bbox (allow small leakage).
        def _mask_mostly_inside_bbox(mask_bin, x1, y1, x2, y2, tol_outside_ratio=0.02):
            if mask_bin is None:
                return False
            if mask_bin.dtype != np.uint8:
                mask_bin = mask_bin.astype(np.uint8)

            h_m, w_m = mask_bin.shape[:2]
            x1c = int(max(0, min(w_m - 1, x1)))
            x2c = int(max(0, min(w_m - 1, x2)))
            y1c = int(max(0, min(h_m - 1, y1)))
            y2c = int(max(0, min(h_m - 1, y2)))
            if x2c <= x1c or y2c <= y1c:
                return False

            mask_bool = (mask_bin > 0)
            total = int(mask_bool.sum())
            if total <= 0:
                return False

            inside = int(mask_bool[y1c:y2c, x1c:x2c].sum())
            outside = total - inside
            return (outside / max(1, total)) <= float(tol_outside_ratio)

        # Maintain frame index ourselves (avoid cap.get(CAP_PROP_POS_FRAMES) cost)
        if not hasattr(self, "current_frame_idx") or self.current_frame_idx is None:
            self.current_frame_idx = -1

        # ---------------------------------------------------
        t0 = time.perf_counter()
        success, frame = self.cap.read()
        t1 = time.perf_counter()
        if not success or frame is None:
            return None, None

        self.current_frame_idx += 1

        # Resize to work resolution
        t2 = time.perf_counter()
        if self.scale_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (self.scale_width, self.scale_height),
                interpolation=cv2.INTER_AREA,
            )
        t3 = time.perf_counter()

        # NOTE: vis_frame is the clean frame (no black mask) for visualization/writer
        vis_frame = frame.copy()

        h, w = frame.shape[:2]

        # Black rectangle to cover logo on 'frame' (used for gray/of/detection)
        t4 = time.perf_counter()
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

        # Gray for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t5 = time.perf_counter()

        # YOLO detection interval
        det_interval = int(getattr(self, "det_interval", 5))
        run_detection = (self.current_frame_idx % det_interval == 0)

        # -------- STEP0: Optical flow delta estimation --------
        t6 = time.perf_counter()
        delta_x = float("nan")
        delta_y = float("nan")
        has_delta = False
        num_inliers = 0
        inlier_ratio = 0.0

        if self.of_tracking and self.of_prev_gray is not None and self.of_prev_pts is not None:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.of_prev_gray,
                gray,
                self.of_prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0001),
            )

            if next_pts is not None and status is not None:
                status = status.reshape(-1)
                prev_pts_flat = self.of_prev_pts.reshape(-1, 2)
                next_pts_flat = next_pts.reshape(-1, 2)

                good_mask = (status == 1)
                pts1 = prev_pts_flat[good_mask]
                pts2 = next_pts_flat[good_mask]

                if pts1.shape[0] >= self.of_min_track_points:
                    ok, dx, dy, inlier_ratio, inlier_mask, num_inliers = self.estimate_translation_ransac(
                        pts1, pts2,
                        max_iters=self.of_ransac_max_iters,
                        thresh=self.of_ransac_threshold,
                    )
                    if ok and inlier_ratio >= self.of_min_inlier_ratio and num_inliers > self.of_min_track_points:
                        delta_x = dx
                        delta_y = dy
                        has_delta = True

                        # Keep inliers only
                        pts2_inlier = pts2[inlier_mask]
                        self.of_prev_pts = pts2_inlier.reshape(-1, 1, 2)
                        self.of_prev_gray = gray  # avoid gray.copy()

                        if self.debug_mode:
                            for p in pts2_inlier:
                                x, y = p
                                cv2.circle(
                                    vis_frame,
                                    (int(round(x)), int(round(y))),
                                    2,
                                    (0, 255, 255),
                                    -1,
                                    lineType=cv2.LINE_AA,
                                )
                    else:
                        self.of_tracking = False
                        self.of_prev_gray = None
                        self.of_prev_pts = None

                    if self.debug_mode:
                        cv2.putText(vis_frame, str(num_inliers), (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, str(inlier_ratio), (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, str(pts1.shape[0]), (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    self.of_tracking = False
                    self.of_prev_gray = None
                    self.of_prev_pts = None
            else:
                self.of_tracking = False
                self.of_prev_gray = None
                self.of_prev_pts = None
        t7 = time.perf_counter()

        # Write temp (clean frame)
        t8 = time.perf_counter()
        if self.temp_writer is not None:
            self.temp_writer.write(vis_frame)
        t9 = time.perf_counter()

        # -------- STEP1: YOLO detection --------
        t10 = time.perf_counter()
        selected_xywh = []
        selected_confs = []
        selected_masks = []
        mask_union = None

        if run_detection:
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

            boxes = result.boxes
            masks = getattr(result, "masks", None)

            if boxes is not None and boxes.xywh is not None and self.target_class_id is not None:
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().numpy()

                mask_data = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    mask_data = masks.data.cpu().numpy()  # (N, Hm, Wm)

                for idx, ((cx, cy, w_box, h_box), conf, cls_id) in enumerate(zip(xywh, confs, cls_ids)):
                    if cls_id == self.target_class_id:
                        selected_xywh.append((float(cx), float(cy), float(w_box), float(h_box)))
                        selected_confs.append(float(conf))

                        if mask_data is not None and idx < mask_data.shape[0]:
                            m = mask_data[idx]
                            if m.shape != gray.shape:
                                m_resized = cv2.resize(
                                    m.astype("float32"),
                                    (w, h),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                m_bin = (m_resized > 0.5).astype("uint8")
                            else:
                                m_bin = (m > 0.5).astype("uint8")

                            if mask_union is None:
                                mask_union = m_bin
                            else:
                                mask_union |= m_bin

                            selected_masks.append(m_bin)
                        else:
                            selected_masks.append(None)
        t11 = time.perf_counter()

        num_obj = len(selected_xywh)

        # -------- STEP2: HUD drawing --------
        t12 = time.perf_counter()
        h_img, w_img = vis_frame.shape[:2]
        lock_color = (0, 255, 0)

        base = min(h_img, w_img)
        line_thickness = max(2, base // 400)
        corner_ratio = 0.25
        cross_ratio = 0.12
        circle_ratio = 0.10

        for idx, (cx, cy, w_box, h_box) in enumerate(selected_xywh):
            color = lock_color if idx == 0 else (120, 255, 120)

            x1b = int(cx - w_box / 2.0)
            y1b = int(cy - h_box / 2.0)
            x2b = int(cx + w_box / 2.0)
            y2b = int(cy + h_box / 2.0)

            w_i = max(1, x2b - x1b)
            h_i = max(1, y2b - y1b)
            edge = min(w_i, h_i)

            corner_len = max(4, int(corner_ratio * edge))
            cross_len = max(3, int(cross_ratio * edge))
            circle_r = max(3, int(circle_ratio * edge))

            # corners
            cv2.line(vis_frame, (x1b, y1b), (x1b + corner_len, y1b), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x1b, y1b), (x1b, y1b + corner_len), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x2b, y1b), (x2b - corner_len, y1b), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x2b, y1b), (x2b, y1b + corner_len), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x1b, y2b), (x1b + corner_len, y2b), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x1b, y2b), (x1b, y2b - corner_len), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x2b, y2b), (x2b - corner_len, y2b), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (x2b, y2b), (x2b, y2b - corner_len), color, line_thickness, cv2.LINE_AA)

            # crosshair
            cx_i = int(cx)
            cy_i = int(cy)
            cv2.line(vis_frame, (cx_i - cross_len, cy_i), (cx_i + cross_len, cy_i), color, line_thickness, cv2.LINE_AA)
            cv2.line(vis_frame, (cx_i, cy_i - cross_len), (cx_i, cy_i + cross_len), color, line_thickness, cv2.LINE_AA)
            cv2.circle(vis_frame, (cx_i, cy_i), circle_r, color, 1, cv2.LINE_AA)
        t13 = time.perf_counter()

        # -------- STEP3: Reset optical flow ref on detection frames --------
        # STRICT MODE: Only refresh points when mask exists AND is mostly inside bbox,
        # and use (mask ∩ bbox) as the feature ROI. No bbox fallback.
        t14 = time.perf_counter()
        if run_detection and num_obj == 1:
            cx, cy, w_box, h_box = selected_xywh[0]

            x1 = int(max(0, cx - w_box / 2.0))
            y1 = int(max(0, cy - h_box / 2.0))
            x2 = int(min(w - 1, cx + w_box / 2.0))
            y2 = int(min(h - 1, cy + h_box / 2.0))

            mask_bin = selected_masks[0] if (selected_masks and selected_masks[0] is not None) else None

            # Only proceed when mask is present and reasonable.
            if mask_bin is not None and _mask_mostly_inside_bbox(mask_bin, x1, y1, x2, y2, tol_outside_ratio=0.02):
                # ROI = mask ∩ bbox
                m = (mask_bin > 0).astype(np.uint8) * 255
                bbox_mask = np.zeros_like(gray, dtype=np.uint8)
                bbox_mask[y1:y2, x1:x2] = 255
                mask_for_of = cv2.bitwise_and(m, bbox_mask)

                pts = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=700,
                    qualityLevel=0.001,
                    minDistance=2.0,
                    mask=mask_for_of,
                )

                if pts is not None and pts.shape[0] >= self.of_min_track_points:
                    self.of_prev_gray = gray  # avoid gray.copy()
                    self.of_prev_pts = pts.reshape(-1, 1, 2)
                    self.of_tracking = True
                else:
                    # We attempted a refresh in a "trusted" ROI but failed to get enough points.
                    # Disable OF to avoid using stale/weak points.
                    self.of_prev_gray = None
                    self.of_prev_pts = None
                    self.of_tracking = False
            # else: do NOTHING (no bbox fallback, keep current OF state as-is)
        t15 = time.perf_counter()

        # Debug mask overlay
        t16 = time.perf_counter()
        if self.debug_mode:
            if mask_union is not None and mask_union.any():
                overlay = vis_frame.astype(np.float32)
                color = np.array([0, 255, 0], dtype=np.float32)
                alpha = 0.6
                mask_bool = mask_union.astype(bool)
                overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color * alpha
                vis_frame = overlay.astype(np.uint8)
        t17 = time.perf_counter()

        # Write shown frame
        t18 = time.perf_counter()
        if self.temp_writer_show is not None:
            self.temp_writer_show.write(vis_frame)
        t19 = time.perf_counter()

        # -------- STEP4: Build TrackFrame --------
        t20 = time.perf_counter()
        if hasattr(self, "fps") and self.fps > 1e-6:
            timestamp = float(self.current_frame_idx / self.fps)
        else:
            timestamp = 0.0

        if run_detection and num_obj == 1:
            cx, cy, w_box, h_box = selected_xywh[0]
            conf = selected_confs[0]
            track_frame = TrackFrame(
                timestamp=timestamp,
                frame_idx=self.current_frame_idx,
                cx=cx, cy=cy, w=w_box, h=h_box,
                conf=conf,
                valid=True,
                num_obj=num_obj,
                dx=delta_x, dy=delta_y,
                has_delta=bool(has_delta),
            )
        else:
            track_frame = TrackFrame(
                timestamp=timestamp,
                frame_idx=self.current_frame_idx,
                cx=0.0, cy=0.0, w=0.0, h=0.0,
                conf=0.0,
                valid=False,
                num_obj=num_obj,
                dx=delta_x, dy=delta_y,
                has_delta=bool(has_delta),
            )

        self.track_results.append(track_frame)
        t21 = time.perf_counter()

        # -------- Print profiling --------
        if profile and (self.current_frame_idx % profile_every == 0):
            total_ms = _ms(t_begin, t21)
            fps_est = 1000.0 / max(1e-6, total_ms)
            print(
                f"[next_frame] idx={self.current_frame_idx} run_det={int(run_detection)} "
                f"total={total_ms:.1f}ms ({fps_est:.1f} FPS) | "
                f"read={_ms(t0,t1):.1f} resize={_ms(t2,t3):.1f} "
                f"prep(gray+mask)={_ms(t4,t5):.1f} "
                f"of={_ms(t6,t7):.1f} w1={_ms(t8,t9):.1f} "
                f"yolo={_ms(t10,t11):.1f} hud={_ms(t12,t13):.1f} "
                f"gftt={_ms(t14,t15):.1f} dbgmask={_ms(t16,t17):.1f} "
                f"w2={_ms(t18,t19):.1f} track={_ms(t20,t21):.1f}"
            )

        return vis_frame, track_frame



    def reset_optical_flow_state(self) -> None:
        """Reset optical flow tracking and related state."""
        self.of_tracking = False
        self.of_prev_gray = None
        self.of_prev_pts = None

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
    smooth_factor: float = 0.5,  # 预留参数，可调平滑/约束权重
    debug: bool = False,
) -> List[CropFrame]:
    """
    规划裁切轨迹（odom + 弱 BBOX 全局约束的一次优化版）.

    模型（对 x/y 各做一次一维优化）:
      - 变量：每一帧的裁切中心 x_i / y_i
      - 里程计约束（有光流时，高权重）:
          对于 has_delta[i] = True:
              r_odom_x = (x_i - x_{i-1}) - dx[i]
              r_odom_y = (y_i - y_{i-1}) - dy[i]
      - 无 odom 时的退化约束:
          * 帧间相等约束（零速度先验）:
              对于 has_delta[i] = False:
                  r_eq_x = (x_i - x_{i-1})
                  r_eq_y = (y_i - y_{i-1})
          * 二阶速度平滑（类似之前二阶差分）:
              对于 has_delta[i] = False 且 1 <= i <= n-2:
                  r_acc_x = x_{i-1} - 2 x_i + x_{i+1}
                  r_acc_y = y_{i-1} - 2 y_i + y_{i+1}
      - 全局 BBOX 约束（弱，全局约束权重很小）:
          对于 valid_det[i] = True:
              r_det_x = x_i - cx_det[i]
              r_det_y = y_i - cy_det[i]
      - 轻微的“居中”先验（避免完全没检测的帧长期漂移）:
          r_center_x = x_i - W/2
          r_center_y = y_i - H/2
      - 第 0 帧有一个较强的 anchor，把整体 gauge 固定住。

    最终是一个稀疏带状系统，对 x, y 各解一次。
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

    # =========================
    # 1. 构建/求解一维优化
    # =========================

    # 权重可以根据经验再调：
    odom_w = 5.0                     # 里程计约束权重（主导平滑）
    det_w = 0.001                     # BBOX 全局约束权重（很小，很弱）
    center_w = 0.00001                 # 轻微“居中”先验
    anchor_w = 0.5                     # 第 0 帧 anchor，固定整体偏移

    # NEW: 无 odom 时的零速度 + 二阶平滑权重（跟 smooth_factor 挂钩）
    eq_w = odom_w * 0.2                # (x_i - x_{i-1})^2, 没有 odom 时的替代约束
    acc_w_base = odom_w * 0.1          # 二阶差分的基准权重
    acc_w = 0 # acc_w_base * float(np.clip(smooth_factor, 0.0, 2.0))

    def solve_1d(
        det_vals: np.ndarray,
        valid_mask: np.ndarray,
        delta: np.ndarray,
        img_center: float,
    ) -> np.ndarray:
        """对一维轨迹做 odom + 无 odom 时退化平滑 + 弱 det + 居中先验的最小二乘优化."""
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        # 我们允许最多到二阶带宽：offsets = -2, -1, 0, 1, 2
        diag = np.zeros(n, dtype=float)
        lower1 = np.zeros(n - 1, dtype=float)  # offset -1
        upper1 = np.zeros(n - 1, dtype=float)  # offset +1
        lower2 = np.zeros(n - 2, dtype=float)  # offset -2
        upper2 = np.zeros(n - 2, dtype=float)  # offset +2
        rhs = np.zeros(n, dtype=float)

        # --- 1) 里程计约束（有 odom 的帧间） ---
        # 对 has_delta[i] == True，构造 (x_i - x_{i-1} - delta[i])^2
        for i in range(1, n):
            if not has_delta_arr[i]:
                continue
            d = delta[i]
            if not np.isfinite(d):
                continue
            w = odom_w
            # A_row = [ ... -1 (i-1), +1 (i) ... ], 观测值 = d
            diag[i - 1] += w
            diag[i] += w
            lower1[i - 1] += -w
            upper1[i - 1] += -w
            rhs[i - 1] += -w * d
            rhs[i] += w * d

        # --- 2) 无 odom 的“帧间相等 + 二次速度平滑” ---
        for i in range(1, n):
            if has_delta_arr[i]:
                continue  # 有 odom 的地方不需要零速度约束

            # 2.1 零速度先验: (x_i - x_{i-1})^2
            if np.isfinite(delta[i]) or True:
                w = eq_w
                diag[i - 1] += w
                diag[i] += w
                lower1[i - 1] += -w
                upper1[i - 1] += -w
                # rhs 不变（观测值 = 0）

            # 2.2 二阶差分: (x_{i-1} - 2 x_i + x_{i+1})^2, 需要 i-1, i, i+1 都存在
            if 1 <= i <= n - 2:
                w2 = acc_w
                # A_row = [ ..., 1 (i-1), -2 (i), 1 (i+1), ...]
                # A^T A = w * [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
                # 对应到对角与 -1, +1, -2, +2 带
                # diag
                diag[i - 1] += w2 * 1.0
                diag[i] += w2 * 4.0
                diag[i + 1] += w2 * 1.0
                # 相邻带 (i-1, i) & (i, i+1)
                lower1[i - 1] += w2 * (-2.0)
                upper1[i - 1] += w2 * (-2.0)
                # (i-1, i+1) 用 offset ±2
                lower2[i - 1] += w2 * 1.0
                upper2[i - 1] += w2 * 1.0
                # rhs 不变（观测值 = 0）

        # --- 3) BBOX 全局约束（弱） ---
        for i in range(n):
            if not valid_mask[i]:
                continue
            v = det_vals[i]
            if not np.isfinite(v):
                continue
            w = det_w
            # 残差: (x_i - v)^2
            diag[i] += w
            rhs[i] += w * v

        # --- 4) 轻微居中先验（所有帧都被轻轻拉向画面中心） ---
        for i in range(n):
            w = center_w
            diag[i] += w
            rhs[i] += w * img_center

        # --- 5) 第 0 帧强 anchor，固定 gauge ---
        # 这里不一定用 det，优先 det，其次用图像中心
        if valid_mask[0] and np.isfinite(det_vals[0]):
            a_val = det_vals[0]
        else:
            a_val = img_center
        diag[0] += anchor_w
        rhs[0] += anchor_w * a_val

        # --- 6) 构建稀疏带状矩阵并求解 ---
        diagonals = [lower2, lower1, diag, upper1, upper2]
        offsets = [-2, -1, 0, 1, 2]

        A = diags(
            diagonals=diagonals,
            offsets=offsets,
            format="csc",
        )

        x_opt = spsolve(A, rhs)
        return np.asarray(x_opt, dtype=float)

    # 分别优化 x / y
    center_x = solve_1d(
        det_vals=cx_det,
        valid_mask=valid_det,
        delta=dx_arr,
        img_center=W / 2.0,
    )
    center_y = solve_1d(
        det_vals=cy_det,
        valid_mask=valid_det,
        delta=dy_arr,
        img_center=H / 2.0,
    )

    # =========================
    # 2. 根据 max_crop_ratio 决定裁切宽高
    # =========================
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
                clamp=False,  # 仍然不在规划阶段做 clamp，黑边交给导出函数处理
            )
        )

    # =========================
    # 3. Debug 可视化
    # =========================
    if debug:
        try:
            import matplotlib.pyplot as plt

            t = timestamps

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.title("Crop Center X: optimized vs detections")
            plt.plot(t, center_x, "-", label="optimized_x")
            plt.axhline(W / 2.0, linestyle="--", label="image_center")
            if np.any(valid_det):
                plt.scatter(
                    t[valid_det],
                    cx_det[valid_det],
                    s=10,
                    marker="x",
                    label="bbox_cx",
                )
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.title("Crop Center Y: optimized vs detections")
            plt.plot(t, center_y, "-", label="optimized_y")
            plt.axhline(H / 2.0, linestyle="--", label="image_center")
            if np.any(valid_det):
                plt.scatter(
                    t[valid_det],
                    cy_det[valid_det],
                    s=10,
                    marker="x",
                    label="bbox_cy",
                )
            plt.legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[planning debug] matplotlib not available or failed: {e}")

    return crop_frames

from typing import List, Optional
import numpy as np

def refine_crop_traj_with_lowfreq(
    crop_frames: List["CropFrame"],
    track_result: List["TrackFrame"],
    keep_lowfreq: float = 0.35,
    inner_ratio: float = 0.55,
    tau_sec: float = 1.2,
    debug: bool = False,
) -> List["CropFrame"]:
    """
    Post-process crop centers to preserve low-frequency 'handheld composition' drift while
    still suppressing high-frequency jitter.

    Concept:
      - Input crop_frames contain a "strong stabilized" crop center trajectory (x_stab/y_stab).
      - Build a low-frequency trajectory (x_low/y_low) by a first-order low-pass filter.
      - Blend: x_mix = (1-k)*x_stab + k*x_low  (k = keep_lowfreq)
      - Project to a feasible interval that keeps the target inside an "inner box":
          |cx_det - x_out| <= margin_x,  margin_x = 0.5*crop_width*inner_ratio
        Same for y.

    Parameters:
      keep_lowfreq: [0,1], larger keeps more low-frequency drift (less "dead center").
      inner_ratio:  (0,1], how much of half crop size is allowed for target movement.
      tau_sec:      low-pass time constant, larger => slower composition drift.
      debug:        optional matplotlib plot.

    Notes:
      - If detection is missing in a frame, we skip projection (only blending applies).
      - We do NOT clamp to image bounds here, consistent with your pipeline (black bars handled later).
    """
    if not crop_frames or not track_result:
        return crop_frames

    n = min(len(crop_frames), len(track_result))
    if n <= 1:
        return crop_frames

    # Sanitize params
    keep_lowfreq = float(np.clip(keep_lowfreq, 0.0, 1.0))
    inner_ratio = float(np.clip(inner_ratio, 1e-3, 1.0))
    tau_sec = float(max(tau_sec, 1e-4))

    # Extract arrays
    t = np.array([track_result[i].timestamp for i in range(n)], dtype=float)
    valid = np.array([bool(getattr(track_result[i], "valid", False)) for i in range(n)], dtype=bool)
    cx_det = np.array([float(getattr(track_result[i], "cx", np.nan)) for i in range(n)], dtype=float)
    cy_det = np.array([float(getattr(track_result[i], "cy", np.nan)) for i in range(n)], dtype=float)

    x_stab = np.array([float(crop_frames[i].crop_center_x) for i in range(n)], dtype=float)
    y_stab = np.array([float(crop_frames[i].crop_center_y) for i in range(n)], dtype=float)

    # Crop size (can be per-frame, but typically constant in your current implementation)
    cw = np.array([float(crop_frames[i].crop_width) for i in range(n)], dtype=float)
    ch = np.array([float(crop_frames[i].crop_height) for i in range(n)], dtype=float)
    margin_x = 0.5 * cw * inner_ratio
    margin_y = 0.5 * ch * inner_ratio

    # ----- 1) Low-pass filter to get low-frequency composition track -----
    # First-order IIR low-pass: y[i] = (1-a)*y[i-1] + a*x[i]
    # a = 1 - exp(-dt/tau)
    x_low = np.empty_like(x_stab)
    y_low = np.empty_like(y_stab)
    x_low[0] = x_stab[0]
    y_low[0] = y_stab[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0 / 30.0  # fallback
        a = 1.0 - float(np.exp(-dt / tau_sec))
        x_low[i] = (1.0 - a) * x_low[i - 1] + a * x_stab[i]
        y_low[i] = (1.0 - a) * y_low[i - 1] + a * y_stab[i]

    # ----- 2) Blend stabilized track with low-frequency track -----
    x_mix = (1.0 - keep_lowfreq) * x_stab + keep_lowfreq * x_low
    y_mix = (1.0 - keep_lowfreq) * y_stab + keep_lowfreq * y_low

    # ----- 3) Project to keep target inside the inner box (if detection valid) -----
    x_out = x_mix.copy()
    y_out = y_mix.copy()

    for i in range(n):
        if not valid[i]:
            continue
        if not (np.isfinite(cx_det[i]) and np.isfinite(cy_det[i])):
            continue
        mx = float(margin_x[i])
        my = float(margin_y[i])
        # Feasible interval: [det - margin, det + margin]
        lo_x, hi_x = cx_det[i] - mx, cx_det[i] + mx
        lo_y, hi_y = cy_det[i] - my, cy_det[i] + my
        x_out[i] = float(np.clip(x_out[i], lo_x, hi_x))
        y_out[i] = float(np.clip(y_out[i], lo_y, hi_y))

    # ----- 4) Write back to CropFrame list (keep other fields unchanged) -----
    out_frames: List["CropFrame"] = []
    for i in range(n):
        cf = crop_frames[i]
        out_frames.append(
            type(cf)(
                timestamp=cf.timestamp,
                frame_idx=cf.frame_idx,
                crop_center_x=float(x_out[i]),
                crop_center_y=float(y_out[i]),
                crop_width=cf.crop_width,
                crop_height=cf.crop_height,
                scale=getattr(cf, "scale", 1.0),
                clamp=getattr(cf, "clamp", False),
            )
        )

    # Pass-through any remaining frames (if lengths differ)
    for i in range(n, len(crop_frames)):
        out_frames.append(crop_frames[i])

    # ----- Optional debug -----
    if debug:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.title("Refine crop centers: stab vs lowfreq vs out")
            plt.plot(t[:n], x_stab, "-", label="x_stab")
            plt.plot(t[:n], x_low, "-", label="x_low")
            plt.plot(t[:n], x_out, "-", label="x_out")
            if np.any(valid):
                plt.scatter(t[valid], cx_det[valid], s=10, marker="x", label="cx_det")
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.title("Refine crop centers: stab vs lowfreq vs out (Y)")
            plt.plot(t[:n], y_stab, "-", label="y_stab")
            plt.plot(t[:n], y_low, "-", label="y_low")
            plt.plot(t[:n], y_out, "-", label="y_out")
            if np.any(valid):
                plt.scatter(t[valid], cy_det[valid], s=10, marker="x", label="cy_det")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[refine debug] matplotlib not available or failed: {e}")

    return out_frames

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

import os
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union

import av
import numpy as np
from av.audio.resampler import AudioResampler

BorderValue = Union[int, Tuple[int, int, int]]


def _has_encoder(name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:
        return False


def _encoder_from_codec_name(codec_name: str) -> str:
    name = (codec_name or "").lower()
    if name in ("h264", "avc1"):
        return "libx264"
    if name in ("hevc", "h265"):
        return "libx265"
    if name == "vp9":
        return "libvpx-vp9"
    if name == "av1":
        return "libaom-av1"
    return "libx264"

def _warp_translate_int_fast(
    img: np.ndarray,
    dx: float,
    dy: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Fast integer-translate warp (nearest / no interpolation).
    Equivalent to warpAffine with dx,dy rounded to int, BORDER_CONSTANT black.
    """
    H, W, C = img.shape
    assert C == 3

    ix = int(round(dx))
    iy = int(round(dy))

    # output pixel u maps to source u - dx  -> source window starts at -dx
    src_x0 = -ix
    src_y0 = -iy
    src_x1 = src_x0 + out_w
    src_y1 = src_y0 + out_h

    out = np.zeros((out_h, out_w, 3), dtype=img.dtype)

    sx0 = max(0, src_x0)
    sy0 = max(0, src_y0)
    sx1 = min(W, src_x1)
    sy1 = min(H, src_y1)

    if sx1 <= sx0 or sy1 <= sy0:
        return out

    dx0 = sx0 - src_x0
    dy0 = sy0 - src_y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out

def _warp_translate_bilinear(
    img: np.ndarray,
    dx: float,
    dy: float,
    out_w: int,
    out_h: int,
    border_value: BorderValue = (0, 0, 0),
) -> np.ndarray:
    """
    Equivalent to:
      cv2.warpAffine(img, [[1,0,dx],[0,1,dy]], (out_w,out_h),
                    flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=border_value)

    img: HxWx3 uint8 (bgr24)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"_warp_translate_bilinear expects HxWx3, got {img.shape}")
    H, W, _ = img.shape
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"invalid output size: {out_w}x{out_h}")

    u = np.arange(out_w, dtype=np.float32)
    v = np.arange(out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # dst(u,v) = src(u - dx, v - dy)
    x = uu - np.float32(dx)
    y = vv - np.float32(dy)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)

    def in_bounds(xx, yy):
        return (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)

    m00 = in_bounds(x0, y0)
    m10 = in_bounds(x1, y0)
    m01 = in_bounds(x0, y1)
    m11 = in_bounds(x1, y1)

    out = np.zeros((out_h, out_w, 3), dtype=np.float32)

    def gather(xx, yy, mask):
        xx_clip = np.clip(xx, 0, W - 1)
        yy_clip = np.clip(yy, 0, H - 1)
        val = img[yy_clip, xx_clip].astype(np.float32)
        val *= mask[..., None].astype(np.float32)
        return val

    I00 = gather(x0, y0, m00)
    I10 = gather(x1, y0, m10)
    I01 = gather(x0, y1, m01)
    I11 = gather(x1, y1, m11)

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    out += I00 * w00[..., None]
    out += I10 * w10[..., None]
    out += I01 * w01[..., None]
    out += I11 * w11[..., None]

    any_in = m00 | m10 | m01 | m11
    if isinstance(border_value, tuple):
        bv = np.array(border_value, dtype=np.float32).reshape(1, 1, 3)
        out[~any_in] = bv
    else:
        out[~any_in] = float(border_value)

    return np.clip(out, 0, 255).astype(np.uint8)


# def export_stabilized_video(
#     input_video_path: str,
#     crop_frames: List["CropFrame"],
#     output_video_path: str,
#     work_width: int,
#     work_height: int,
#     progress_cb: Optional[Callable[[float], None]] = None,
#     add_brand_watermark: bool = False,  # 保留参数以兼容旧接口（本实现不做水印）
# ) -> None:
#     """
#     Drop-in exporter:
#       - PyAV decode + encode
#       - Per-frame translation warp (linear + constant border black)
#       - Audio: try stream copy; fallback to AAC (with resample); if AAC not available -> no audio
#       - FPS uses rational rate
#       - IMPORTANT FIX: DO NOT trust in_video.frames for clamping (can be wrong)
#       - Bitrate: if input has bitrate, try to keep similar bitrate for output
#     """
#     if not crop_frames:
#         raise RuntimeError("export_stabilized_video: crop_frames is empty")
#     if work_width <= 0 or work_height <= 0:
#         raise RuntimeError(f"export_stabilized_video: invalid work size {work_width}x{work_height}")

#     os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

#     in_container = None
#     out_container = None

#     try:
#         in_container = av.open(input_video_path)

#         in_video = next((s for s in in_container.streams if s.type == "video"), None)
#         if in_video is None:
#             raise RuntimeError(f"export_stabilized_video: no video stream: {input_video_path}")

#         in_audio = next((s for s in in_container.streams if s.type == "audio"), None)

#         src_w = int(in_video.codec_context.width or 0)
#         src_h = int(in_video.codec_context.height or 0)
#         if src_w <= 0 or src_h <= 0:
#             raise RuntimeError(
#                 f"export_stabilized_video: input video has invalid size {src_w}x{src_h} (path={input_video_path})"
#             )

#         # FPS as Fraction/Rational (PyAV prefers this)
#         fps_rate = in_video.average_rate
#         if fps_rate is None:
#             fps_rate = Fraction(30, 1)
#         try:
#             fps_float = float(fps_rate)
#         except Exception:
#             fps_rate = Fraction(30, 1)
#             fps_float = 30.0
#         if (not np.isfinite(fps_float)) or fps_float <= 1e-3:
#             fps_rate = Fraction(30, 1)
#             fps_float = 30.0

#         # ✅ FIX: trust plan length, NOT in_video.frames
#         total_frames = len(crop_frames)
#         if total_frames <= 0:
#             raise RuntimeError("export_stabilized_video: no frames to export (total_frames == 0)")

#         # Work->full scaling
#         sx = src_w / float(work_width)
#         sy = src_h / float(work_height)

#         # Output size from first crop frame (same semantics as your original)
#         first_cf = crop_frames[0]
#         out_w = int(round(float(first_cf.crop_width) * sx))
#         out_h = int(round(float(first_cf.crop_height) * sy))
#         out_w = min(out_w, src_w)
#         out_h = min(out_h, src_h)

#         # Even sizes
#         if out_w % 2 == 1:
#             out_w -= 1
#         if out_h % 2 == 1:
#             out_h -= 1
#         if out_w <= 0 or out_h <= 0:
#             raise RuntimeError(f"export_stabilized_video: invalid output size {out_w}x{out_h}")

#         out_container = av.open(output_video_path, mode="w")

#         # --- Video stream (best-effort keep codec family) ---
#         in_codec_name = (in_video.codec_context.name or in_video.codec.name or "").lower()
#         encoder_name = _encoder_from_codec_name(in_codec_name)
#         if not _has_encoder(encoder_name):
#             encoder_name = "libx264"

#         try:
#             out_v = out_container.add_stream(encoder_name, rate=fps_rate)
#         except Exception:
#             out_v = out_container.add_stream("libx264", rate=fps_rate)

#         out_v.width = out_w
#         out_v.height = out_h
#         out_v.pix_fmt = "yuv420p"

#         # ✅ Bitrate policy: if input has bitrate, try to keep it similar (closer file size)
#         in_br = int(getattr(in_video.codec_context, "bit_rate", 0) or 0)
#         if in_br > 0:
#             try:
#                 out_v.bit_rate = in_br
#                 # keep preset only; avoid CRF overriding ABR intent
#                 out_v.options = {"preset": "medium"}
#             except Exception:
#                 # fallback to CRF
#                 out_v.options = {"crf": "18", "preset": "medium"}
#         else:
#             out_v.options = {"crf": "18", "preset": "medium"}

#         # --- Audio stream: copy -> AAC fallback(with resample) -> no audio ---
#         out_a = None
#         audio_copy = False
#         audio_resampler: Optional[AudioResampler] = None

#         if in_audio is not None:
#             # Try stream copy
#             try:
#                 out_a = out_container.add_stream(template=in_audio)
#                 audio_copy = True
#             except Exception:
#                 out_a = None
#                 audio_copy = False

#             # AAC fallback (stable)
#             if out_a is None:
#                 if _has_encoder("aac"):
#                     out_a = out_container.add_stream("aac", rate=48000)
#                     out_a.bit_rate = 192_000
#                     try:
#                         out_a.layout = "stereo"
#                     except Exception:
#                         pass
#                     audio_resampler = AudioResampler(format="fltp", layout="stereo", rate=48000)
#                 else:
#                     out_a = None  # no audio

#         export_duration_sec = float(total_frames) / float(fps_float)

#         dst_cx = out_w / 2.0
#         dst_cy = out_h / 2.0

#         frame_idx = 0
#         video_done = False
#         audio_done = (in_audio is None or out_a is None)

#         streams = [in_video]
#         if in_audio is not None:
#             streams.append(in_audio)

#         for packet in in_container.demux(streams):
#             stype = packet.stream.type

#             # ---- Audio ----
#             if stype == "audio" and (not audio_done) and (out_a is not None):
#                 if packet.pts is not None and packet.time_base is not None:
#                     if float(packet.pts * packet.time_base) >= export_duration_sec:
#                         audio_done = True
#                         continue

#                 if audio_copy:
#                     try:
#                         packet.stream = out_a
#                         out_container.mux(packet)
#                     except Exception:
#                         # Switch to AAC re-encode if copy fails
#                         audio_copy = False

#                         if out_a.codec.name != "aac":
#                             if _has_encoder("aac"):
#                                 out_a = out_container.add_stream("aac", rate=48000)
#                                 out_a.bit_rate = 192_000
#                                 try:
#                                     out_a.layout = "stereo"
#                                 except Exception:
#                                     pass
#                                 audio_resampler = AudioResampler(format="fltp", layout="stereo", rate=48000)
#                             else:
#                                 out_a = None
#                                 audio_done = True
#                                 continue

#                         for aframe in packet.decode():
#                             if aframe.pts is not None and aframe.time_base is not None:
#                                 if float(aframe.pts * aframe.time_base) >= export_duration_sec:
#                                     audio_done = True
#                                     break

#                             if audio_resampler is not None:
#                                 res = audio_resampler.resample(aframe)
#                                 if res is None:
#                                     continue
#                                 if not isinstance(res, list):
#                                     res = [res]
#                                 for rf in res:
#                                     for opkt in out_a.encode(rf):
#                                         out_container.mux(opkt)
#                             else:
#                                 for opkt in out_a.encode(aframe):
#                                     out_container.mux(opkt)
#                 else:
#                     if out_a is None:
#                         audio_done = True
#                         continue
#                     for aframe in packet.decode():
#                         if aframe.pts is not None and aframe.time_base is not None:
#                             if float(aframe.pts * aframe.time_base) >= export_duration_sec:
#                                 audio_done = True
#                                 break

#                         if audio_resampler is not None:
#                             res = audio_resampler.resample(aframe)
#                             if res is None:
#                                 continue
#                             if not isinstance(res, list):
#                                 res = [res]
#                             for rf in res:
#                                 for opkt in out_a.encode(rf):
#                                     out_container.mux(opkt)
#                         else:
#                             for opkt in out_a.encode(aframe):
#                                 out_container.mux(opkt)

#                 if video_done and audio_done:
#                     break
#                 continue

#             # ---- Video ----
#             if stype != "video":
#                 continue

#             if video_done:
#                 if audio_done:
#                     break
#                 continue

#             for frame in packet.decode():
#                 if frame_idx >= total_frames:
#                     video_done = True
#                     break

#                 img = frame.to_ndarray(format="bgr24")

#                 cf = crop_frames[frame_idx]
#                 cx_full = float(cf.crop_center_x) * sx
#                 cy_full = float(cf.crop_center_y) * sy
#                 if not (np.isfinite(cx_full) and np.isfinite(cy_full)):
#                     cx_full = src_w / 2.0
#                     cy_full = src_h / 2.0

#                 dx = dst_cx - cx_full
#                 dy = dst_cy - cy_full
                
#                  # _warp_translate_int_fast( #TBD
#                 out_img = _warp_translate_bilinear(
#                     img=img,
#                     dx=dx,
#                     dy=dy,
#                     out_w=out_w,
#                     out_h=out_h,
#                     #border_value=(0, 0, 0),
#                 )

#                 out_frame = av.VideoFrame.from_ndarray(out_img, format="bgr24")
#                 out_frame.pts = frame_idx  # CFR

#                 for opkt in out_v.encode(out_frame):
#                     out_container.mux(opkt)

#                 frame_idx += 1
#                 if progress_cb is not None:
#                     progress_cb(100.0 * frame_idx / float(max(1, total_frames)))

#             if video_done and audio_done:
#                 break

#         # Flush encoders
#         for opkt in out_v.encode():
#             out_container.mux(opkt)

#         if out_a is not None and (not audio_copy):
#             for opkt in out_a.encode():
#                 out_container.mux(opkt)

#         # (可选) 你想定位体积差异时，可以临时打开这几行
#         print("exported frames:", frame_idx)
#         print("out size:", out_w, out_h, "fps:", float(fps_rate))
#         print("audio:", "enabled" if out_a is not None else "none", "copy" if audio_copy else "re-enc/none")
#         print("input bitrate:", in_br)

#     except Exception as e:
#         raise RuntimeError(f"export_stabilized_video: error: {e}") from e
#     finally:
#         if in_container is not None:
#             try:
#                 in_container.close()
#             except Exception:
#                 pass
#         if out_container is not None:
#             try:
#                 out_container.close()
#             except Exception:
#                 pass


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