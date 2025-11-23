import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import sys
import torch

def resource_path(relative_path: str) -> str:
    """兼容普通运行和 PyInstaller 打包后的资源路径."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller 打包后一切会被解压到这个临时目录
        base_path = sys._MEIPASS
    else:
        # 开发时：以当前文件所在目录为基准（即 core/）
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class AirSteadyEngine:
    """
    AirSteady 核心算法封装：
    - 使用 YOLO 的 track() 做多目标跟踪，基于 track_id 选择单一主体
    - 使用 EMA 对主体中心做平滑
    - 根据平滑中心裁剪得到稳像画面，并打水印
    - 在线预览时顺便把每帧的“平滑中心”缓存起来
    - 导出时不再做 YOLO 跟踪，直接用缓存数据 + 原视频帧做裁剪
    - 导出时：
        * 帧率保持与原视频一致
        * 可以选择导出分辨率
        * 把原始音频一并拷贝到新视频中
    """
    

    def __init__(
        self,
        model_path: str = "model.pt",
        crop_ratio: float = 0.7,
        ema_alpha: float = 0.6,
        candidate_names=None,
        lost_threshold_frames: int = 10,
        mode: str = "semi",  # "semi" 半自动; "auto" 全自动（简单自动接上）
    ):
        if candidate_names is None:
            candidate_names = ["airplane", "bird", "kite"]

        if not (0.0 < crop_ratio <= 1.0):
            raise ValueError(f"crop_ratio 必须在 (0, 1]，当前为 {crop_ratio}")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha 必须在 (0, 1]，当前为 {ema_alpha}")

        # 如果没有显式传，就用 core/model/model.pt
        if not os.path.isabs(model_path):
            # 注意这里拼的是 model/<你的文件名>，匹配 core/model 目录
            model_path = resource_path(os.path.join("model", model_path))
        
        print(model_path)

        self.model_path = model_path
        self.crop_ratio = float(crop_ratio)
        self.ema_alpha = float(ema_alpha)
        self.candidate_names = candidate_names
        self.lost_threshold_frames = int(lost_threshold_frames)
        self.mode = mode

        self.tracker_cfg = resource_path(os.path.join("model", "tracker.yaml"))

        # YOLO 模型
        # self.model = YOLO(self.model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        names = self.model.names  # dict: {0: 'person', ...}
        self.candidate_cls_ids = [
            i for i, n in names.items() if n in self.candidate_names
        ]
        if not self.candidate_cls_ids:
            raise RuntimeError(
                f"模型里没有 {self.candidate_names} 这些类别，检查权重是否是 COCO 预训练。"
            )

        # 视频相关
        self.cap = None
        self.video_path = None
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame_idx = 0

        # 跟踪 & 稳像状态
        self.track_history = defaultdict(lambda: [])
        self.target_track_id = None
        self.smoothed_center = None  # EMA 平滑中心 (cx, cy)

        # 丢失状态
        self.lost_frames = 0
        self.is_lost = False

        # 用户点击相关（当前帧归一化坐标）
        self.last_result = None
        self.last_candidate_boxes = []
        self.last_candidate_track_ids = []
        self.pending_click_norm = None  # (x_norm, y_norm)

        # 在线预览过程中，缓存每帧的平滑中心（归一化坐标）
        # frame_idx -> (cx_norm, cy_norm)
        self.crop_cache: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # 视频与状态管理
    # ------------------------------------------------------------------
    def open_video(self, video_path: str):
        """打开视频并初始化尺寸等信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w <= 0 or h <= 0:
            # 回退方案：读一帧看看
            success, frame = cap.read()
            if not success:
                cap.release()
                raise RuntimeError("视频为空。")
            h, w = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.cap = cap
        self.video_path = video_path
        self.fps = float(fps)
        self.total_frames = total_frames
        self.width = w
        self.height = h
        self.current_frame_idx = 0

        # 裁剪窗口尺寸
        self.crop_w = int(self.width * self.crop_ratio)
        self.crop_h = int(self.height * self.crop_ratio)

        # 重置状态（包含清空缓存）
        self.reset_tracking(clear_click=True, clear_cache=True)

    def reset_video_to_start(self):
        """
        视频回到开头：
        - 用于导出时重新从 0 帧开始读取；
        - 不清空 crop_cache，导出要复用预览缓存。
        """
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        # 不清理 cache，只重置跟踪状态（避免导出时再跑 YOLO）
        self.reset_tracking(clear_click=False, clear_cache=False)

    def reset_tracking(self, clear_click: bool = False, clear_cache: bool = False):
        """重置跟踪相关状态"""
        self.track_history.clear()
        self.target_track_id = None
        self.smoothed_center = None
        self.lost_frames = 0
        self.is_lost = False
        self.last_result = None
        self.last_candidate_boxes = []
        self.last_candidate_track_ids = []
        if clear_click:
            self.pending_click_norm = None
        if clear_cache:
            self.crop_cache.clear()

    # ------------------------------------------------------------------
    # 用户交互：点击选择目标
    # ------------------------------------------------------------------
    def set_mode(self, mode: str):
        if mode not in ("semi", "auto"):
            raise ValueError("mode 必须是 'semi' 或 'auto'")
        self.mode = mode

    def set_pending_click(self, x_norm: float, y_norm: float):
        """
        记录用户点击的归一化坐标（0~1），后续在当前帧的检测结果中选择最近的 track_id。
        UI 在收到点击事件后调用。
        """
        self.pending_click_norm = (float(x_norm), float(y_norm))

        # 如果当前已经有检测结果，可以立即根据当前帧的候选框选定目标
        if self.last_candidate_boxes and self.last_candidate_track_ids:
            self._apply_click_to_last_result()

    def _apply_click_to_last_result(self):
        if self.pending_click_norm is None:
            return
        if not self.last_candidate_boxes:
            return

        x_norm, y_norm = self.pending_click_norm
        px = x_norm * self.width
        py = y_norm * self.height

        # 选择与点击位置最近的候选框中心
        min_dist = None
        chosen_tid = None
        chosen_center = None

        for (cx, cy, _, _), tid in zip(self.last_candidate_boxes, self.last_candidate_track_ids):
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                chosen_tid = tid
                chosen_center = (cx, cy)

        if chosen_tid is not None:
            self.target_track_id = chosen_tid
            self.smoothed_center = np.array(chosen_center, dtype=np.float32)
            self.pending_click_norm = None
            # 选定目标后视作不丢失
            self.lost_frames = 0
            self.is_lost = False

    # ------------------------------------------------------------------
    # 内部工具：裁剪 + 丢失状态更新 + 水印
    # ------------------------------------------------------------------
    def _crop_stabilized(self, frame, center):
        """根据平滑后的中心点裁剪 + resize 回原尺寸"""
        cx, cy = center
        w, h = self.width, self.height
        crop_w, crop_h = self.crop_w, self.crop_h

        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)

        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        stabilized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return stabilized

    def _update_lost_state(self, current_center):
        """根据当前帧是否找到 target_track_id 更新丢失状态"""
        if self.target_track_id is None:
            # 未锁定目标，不算丢失
            self.lost_frames = 0
            self.is_lost = False
            return

        if current_center is None:
            self.lost_frames += 1
            if self.lost_frames >= self.lost_threshold_frames:
                self.is_lost = True
        else:
            self.lost_frames = 0
            self.is_lost = False

    def _apply_watermark(self, frame):
        """在稳像画面上打 AirSteady 水印"""
        text = "AirSteady"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        x_text = self.width - text_w - 20
        y_text = 30

        cv2.rectangle(
            frame,
            (x_text - 10, y_text - text_h - 5),
            (x_text + text_w + 10, y_text + 5),
            (0, 0, 0),
            thickness=-1
        )
        cv2.putText(
            frame,
            text,
            (x_text, y_text),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA
        )

    # ------------------------------------------------------------------
    # 单步前向：读取下一帧，做 YOLO + 跟踪 + 稳像（在线预览）
    # ------------------------------------------------------------------
    def next_frame(self):
        """
        读取下一帧并处理（用于在线预览）。
        返回:
            vis_frame: 原始 + 检测/轨迹可视化 (BGR)
            stabilized_frame: 稳像后的帧 (BGR)
            info: dict, 包含:
                - frame_idx
                - total_frames
                - has_target (bool)
                - is_lost (bool)
        若到达视频末尾，返回 (None, None, None)
        """
        if self.cap is None:
            return None, None, None

        success, frame = self.cap.read()
        if not success:
            return None, None, None

        # 当前帧索引
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # YOLO 跟踪（persist=True 保留 track id）
        result = self.model.track(frame, persist=True, verbose=False, device=self.device, tracker=self.tracker_cfg) [0]
        self.last_result = result

        vis_frame = result.plot()

        candidate_boxes = []
        candidate_track_ids = []

        # 取出 tracking 结果
        if result.boxes is not None and result.boxes.id is not None:
            boxes_xywh = result.boxes.xywh.cpu().numpy()  # [x, y, w, h]
            cls_ids = result.boxes.cls.int().cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            # 收集候选类别
            for (x, y, w_box, h_box), cls_id, tid in zip(boxes_xywh, cls_ids, track_ids):
                if cls_id in self.candidate_cls_ids:
                    candidate_boxes.append((float(x), float(y), float(w_box), float(h_box)))
                    candidate_track_ids.append(tid)

            # 可视化每个 track 的短轨迹（可注释）
            for (x, y, w_box, h_box), tid in zip(boxes_xywh, track_ids):
                track = self.track_history[tid]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                pts = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    vis_frame, [pts],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=2
                )

        # 保存候选，供点击使用
        self.last_candidate_boxes = candidate_boxes
        self.last_candidate_track_ids = candidate_track_ids

        # 若有待处理的点击，尝试立即根据当前帧应用
        if self.pending_click_norm is not None:
            self._apply_click_to_last_result()

        # ========= 选择要跟踪的那一只/架 =========
        current_center = None

        if candidate_boxes:
            # 如果还没锁定目标且没有点击，在“全自动模式”下可以自动选一个（面积最大）
            if self.target_track_id is None and self.pending_click_norm is None:
                if self.mode == "auto":
                    areas = [w_box * h_box for (_, _, w_box, h_box) in candidate_boxes]
                    best_idx = int(np.argmax(areas))
                    self.target_track_id = candidate_track_ids[best_idx]

            # 在候选里找到当前锁定 track_id 的框
            if self.target_track_id is not None:
                for (x, y, w_box, h_box), tid in zip(candidate_boxes, candidate_track_ids):
                    if tid == self.target_track_id:
                        current_center = np.array([x, y], dtype=np.float32)
                        # 画出当前锁定目标中心（绿色点）
                        cv2.circle(vis_frame, (int(x), int(y)), 6, (0, 255, 0), -1)
                        break

        # 若当前帧找不到那只/那架：视作丢失
        self._update_lost_state(current_center)

        # ========= EMA 平滑中心 =========
        if current_center is not None:
            if self.smoothed_center is None:
                self.smoothed_center = current_center.copy()
            else:
                self.smoothed_center = (
                    (1 - self.ema_alpha) * self.smoothed_center
                    + self.ema_alpha * current_center
                )

        # ========= 生成稳定画面 + 打水印 =========
        if self.smoothed_center is not None:
            stabilized_frame = self._crop_stabilized(frame, self.smoothed_center)
            self._apply_watermark(stabilized_frame)

            # 把平滑中心存入缓存（归一化）
            cx_norm = float(self.smoothed_center[0] / self.width)
            cy_norm = float(self.smoothed_center[1] / self.height)
            self.crop_cache[self.current_frame_idx] = (cx_norm, cy_norm)
        else:
            # 一开始还没有中心时，就让 stabilized_frame = 原图
            stabilized_frame = frame.copy()

        info = {
            "frame_idx": self.current_frame_idx,
            "total_frames": self.total_frames,
            "has_target": self.target_track_id is not None,
            "is_lost": self.is_lost,
        }

        return vis_frame, stabilized_frame, info

    # ------------------------------------------------------------------
    # 导出：只用缓存的裁剪数据 + 原视频 + 原音频
    # ------------------------------------------------------------------
    def export_video(
        self,
        output_path: str,
        progress_callback=None,
        stop_flag=None,
        click_norm=None,
        export_size=None,  # None 表示原始分辨率，否则 (w, h)
    ):
        """
        导出稳定视频。
        - 不再做 YOLO 跟踪，只是用在线预览时缓存的平滑中心做裁剪。
        - 只写稳像后的单路画面（不再拼接原始画面）。
        - 帧率与原视频一致。
        - 使用 moviepy 把原视频的音频复制到导出视频里。
        - 若缓存不完整（没预览完所有帧），缺失帧会使用“居中裁剪”或原帧。
        """

        if self.video_path is None:
            raise RuntimeError("还没有打开任何视频。")

        # 如果提供了点击坐标，在导出这趟里也记下来（方便部分缓存的情况）
        if click_norm is not None and self.pending_click_norm is None:
            x_norm, y_norm = click_norm
            self.pending_click_norm = (float(x_norm), float(y_norm))

        # 决定导出分辨率
        if export_size is None:
            out_w, out_h = self.width, self.height
        else:
            out_w, out_h = export_size

        # 重新打开视频，从 0 帧开始读取
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法重新打开视频: {self.video_path}")

        self.reset_video_to_start()

        # 临时视频文件（无音频）
        temp_dir = os.path.dirname(output_path) or "."
        temp_video_path = os.path.join(temp_dir, "_airsteady_temp_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (out_w, out_h))

        frame_idx = 0
        while True:
            if stop_flag is not None and callable(stop_flag) and stop_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            # 优先使用缓存的中心
            center = None
            if frame_idx in self.crop_cache:
                cx_norm, cy_norm = self.crop_cache[frame_idx]
                cx = cx_norm * self.width
                cy = cy_norm * self.height
                center = np.array([cx, cy], dtype=np.float32)

            # 没有缓存时：简单居中，不再做 YOLO 跟踪（尊重“不重新预测”的要求）
            if center is not None:
                stabilized_frame = self._crop_stabilized(frame, center)
                self._apply_watermark(stabilized_frame)
            else:
                stabilized_frame = frame.copy()

            # resize 到导出分辨率
            stabilized_resized = cv2.resize(
                stabilized_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )
            out.write(stabilized_resized)

            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, self.total_frames)

        out.release()
        cap.release()

        # 用 moviepy 把原音频合入新视频
        try:
            from moviepy.editor import VideoFileClip

            # 无音频的稳定视频
            video_clip = VideoFileClip(temp_video_path)
            # 原视频（带音频）
            orig_clip = VideoFileClip(self.video_path)

            final_clip = video_clip.set_audio(orig_clip.audio)
            # 保持帧率一致
            final_clip.write_videofile(
                output_path,
                fps=self.fps,
                audio_codec="aac",
                verbose=False,
                logger=None,
            )

            video_clip.close()
            orig_clip.close()
            final_clip.close()
        finally:
            # 删除临时文件
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        # 导出结束后，恢复到当前播放位置（可选）
        self.cap = cv2.VideoCapture(self.video_path)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
