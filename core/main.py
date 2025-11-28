import sys
from enum import Enum, auto
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal, QThread, QObject, Slot, QEvent
from PySide6.QtGui import QPainter, QColor, QFont, QPixmap, QMouseEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolButton,
    QComboBox,
)

from algorithm import TrackEngine  # 你的算法文件
import algorithm
import utils


class AppState(Enum):
    IDLE = auto()
    VIDEO_LOADED = auto()
    TRACKING = auto()
    TRACK_DONE = auto()
    PLANNING = auto()
    PLAN_DONE = auto()
    EXPORTING = auto()


class VideoView(QWidget):
    """
    简化版视频视图：
    - 显示一张 pixmap
    - 支持叠加文字 overlay
    - 支持画目标框（当前未用，可留作扩展）
    - 支持点击，发出归一化坐标
    """
    clicked = Signal(float, float)  # x_norm, y_norm

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._pixmap: QPixmap | None = None
        self._overlay_text: str | None = None
        self._overlay_color = QColor(255, 255, 255)
        self._overlay_bg = QColor(0, 0, 0, 120)
        self._target_rect: QRectF | None = None  # 0~1 归一化坐标
        self._last_draw_rect = QRectF()
        self._overlay_center = True

        self.setMinimumSize(320, 240)

    def set_frame_pixmap(self, pix: QPixmap | None):
        self._pixmap = pix
        self.update()

    def set_overlay(self, text: str | None, center: bool | None, color: QColor | None = None):
        self._overlay_text = text
        if color is not None:
            self._overlay_color = color
        if center is not None:
            self._overlay_center = center
        self.update()

    def clear_overlay(self):
        self._overlay_text = None
        self.update()

    def set_target_rect_norm(self, rect_norm: QRectF | None):
        """
        rect_norm: x, y, w, h 在 [0,1] 范围内
        """
        self._target_rect = rect_norm
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 32, 32))

        # 标题
        painter.setPen(QColor(220, 220, 220))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 20, self._title)

        # 画视频内容区域（留一点上边距给标题）
        content_rect = self.rect().adjusted(5, 25, -5, -5)

        if self._pixmap and not self._pixmap.isNull():
            # 按比例缩放居中
            scaled = self._pixmap.scaled(
                content_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = content_rect.x() + (content_rect.width() - scaled.width()) // 2
            y = content_rect.y() + (content_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # 记录实际绘制区域，用于后续换算坐标
            self._last_draw_rect = QRectF(x, y, scaled.width(), scaled.height())
        else:
            self._last_draw_rect = QRectF()

        # 画目标框（若有）
        if self._target_rect and not self._last_draw_rect.isNull():
            rx = self._target_rect.x()
            ry = self._target_rect.y()
            rw = self._target_rect.width()
            rh = self._target_rect.height()
            x = self._last_draw_rect.x() + rx * self._last_draw_rect.width()
            y = self._last_draw_rect.y() + ry * self._last_draw_rect.height()
            w = rw * self._last_draw_rect.width()
            h = rh * self._last_draw_rect.height()
            painter.setPen(QColor(0, 255, 0))
            painter.drawRect(QRectF(x, y, w, h))

        # 画 overlay 文本
        if self._overlay_text:
            painter.setFont(QFont("Microsoft YaHei", 11))
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(self._overlay_text)
            text_height = metrics.height()

            # 居中放在 content_rect 中间
            tx = content_rect.center().x() - text_width / 2
            if self._overlay_center:
                ty = content_rect.center().y() - text_height / 2
            else:
                ty = content_rect.height() * 0.1

            bg_rect = QRectF(
                tx - 20,
                ty - text_height,
                text_width + 40,
                text_height + 20,
            )
            painter.fillRect(bg_rect, self._overlay_bg)
            painter.setPen(self._overlay_color)
            painter.drawText(
                bg_rect,
                Qt.AlignCenter | Qt.AlignVCenter,
                self._overlay_text,
            )

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and not self._last_draw_rect.isNull():
            if not self._last_draw_rect.contains(event.position()):
                return
            # 将点击位置转换为归一化坐标
            x_norm = (event.position().x() - self._last_draw_rect.x()) / self._last_draw_rect.width()
            y_norm = (event.position().y() - self._last_draw_rect.y()) / self._last_draw_rect.height()
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            self.clicked.emit(x_norm, y_norm)


class ControlPanel(QWidget):
    smoothChanged = Signal(float)     # 镜头稳定程度 [0,1]
    cropChanged = Signal(float)       # 裁切保留比例 keep_ratio [0,1]

    def __init__(self, parent=None):
        super().__init__(parent)

        # 预设（现在放在最上面）
        track_class_label = QLabel("跟踪对象")
        self.track_class_combo = QComboBox()
        self.track_class_combo.addItems(["飞机"])

        # 镜头稳定程度（平滑度）
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(1, 99)
        self.smooth_slider.setValue(50)
        self.smooth_value_label = QLabel("0.50")
        smooth_label = QLabel("镜头稳定程度")
        self.smooth_left_label = QLabel("跟随原片")
        self.smooth_right_label = QLabel("强力稳像")

        # 裁切保留比例
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setRange(40, 100)  # 0.4 ~ 1.0
        self.crop_slider.setValue(80)
        self.crop_value_label = QLabel("0.80")
        crop_label = QLabel("裁切保留比例")
        self.crop_left_label = QLabel("稳定优先（裁切大）")
        self.crop_right_label = QLabel("画质优先（保留多）")

        self.recompute = QPushButton("重新运镜")

        # 布局
        layout = QVBoxLayout(self)
        title = QLabel("参数调节")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(10)
        # Track label.
        layout.addWidget(track_class_label)
        layout.addWidget(self.track_class_combo)

        # Smooth params
        layout.addSpacing(10)
        layout.addWidget(smooth_label)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(self.smooth_left_label)
        smooth_row.addWidget(self.smooth_slider, 1)
        smooth_row.addWidget(self.smooth_right_label)
        layout.addLayout(smooth_row)
        layout.addWidget(self.smooth_value_label)

        # 裁切比例
        layout.addSpacing(10)
        layout.addWidget(crop_label)
        crop_row = QHBoxLayout()
        crop_row.addWidget(self.crop_left_label)
        crop_row.addWidget(self.crop_slider, 1)
        crop_row.addWidget(self.crop_right_label)
        layout.addLayout(crop_row)
        layout.addWidget(self.crop_value_label)

        layout.addSpacing(10)
        layout.addWidget(self.recompute)

        layout.addStretch(1)
        self.setFixedWidth(320)

        # 信号连接
        self.smooth_slider.valueChanged.connect(self._on_smooth_changed)
        self.crop_slider.valueChanged.connect(self._on_crop_changed)

    def _on_smooth_changed(self, v: int):
        alpha = v / 100.0
        self.smooth_value_label.setText(f"{alpha:.2f}")
        self.smoothChanged.emit(alpha)

    def _on_crop_changed(self, v: int):
        keep_ratio = v / 100.0
        self.crop_value_label.setText(f"{keep_ratio:.2f}")
        self.cropChanged.emit(keep_ratio)


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """将 BGR 图像转换为 QPixmap"""
    if frame is None:
        return QPixmap()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class TrackEngineInitWorker(QObject):
    """后台线程里初始化 TrackEngine，用信号把结果丢回主线程。"""
    finished = Signal(object)   # TrackEngine 实例
    error = Signal(str)

    @Slot()
    def run(self):
        try:
            engine = TrackEngine()   # 这里会比较慢：加载 YOLO + warmup
            self.finished.emit(engine)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AirSteady 航迹稳拍 - 内测版")
        self.resize(1400, 800)

        self.state = AppState.IDLE

        # 记录用户最后一次点击的归一化坐标（当前没用，可留作扩展）
        self.last_click_norm = None  # (x_norm, y_norm) 或 None

        # 跟踪阶段计时器
        self.track_play_timer = QTimer(self)
        self.track_play_timer.setInterval(2)  # 2ms 轮询，实际 FPS 由 YOLO 决定
        self.track_play_timer.timeout.connect(self._on_track_obj)

        # 预览阶段计时器
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._on_preview_tick)
        self.preview_cap_raw: cv2.VideoCapture | None = None
        self.preview_cap_steady: cv2.VideoCapture | None = None
        self.preview_total_frames: int = 0
        self.preview_fps: float = 25.0
        self.preview_frame_idx: int = 0

        # 结果缓存
        self.track_engine: TrackEngine | None = None
        self._track_engine_thread: QThread | None = None
        self.track_result_all = None
        self.crop_traj = None
        self.preview_raw_path = ""
        self.preview_steady_path = ""

        self._build_ui()
        self._update_state(AppState.IDLE)

        # 全局拦截空格键
        QApplication.instance().installEventFilter(self)

        # Load Model & init track engine.
        self._start_track_engine_init()

    # --------- 全局键盘拦截，空格控制预览播放 ----------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            # 除 PLAN_DONE 外其他状态一律吃掉空格（不触发按钮）
            if self.state == AppState.PLAN_DONE and self.preview_cap_raw and self.preview_cap_raw.isOpened():
                self._toggle_preview_play_pause()
            return True
        return super().eventFilter(obj, event)

    def _start_track_engine_init(self):
        """在后台线程中初始化 TrackEngine（加载 YOLO 模型）"""
        # 提示一下用户
        self.status.showMessage("算法加载中，约需几秒，请稍后...")
        self.raw_view.set_overlay("算法加载中，约需几秒，请稍后...", center=True)
        self.open_btn.setEnabled(False)

        # 1) 创建线程和 worker
        self._track_engine_thread = QThread(self)
        self._track_engine_worker = TrackEngineInitWorker()
        self._track_engine_worker.moveToThread(self._track_engine_thread)

        # 2) 线程启动时调用 worker.run()
        self._track_engine_thread.started.connect(self._track_engine_worker.run)

        # 3) 加载完成/出错信号
        self._track_engine_worker.finished.connect(self._on_track_engine_ready)
        self._track_engine_worker.error.connect(self._on_track_engine_error)

        # 4) 收尾：退出线程 & 释放 worker / 线程对象
        self._track_engine_worker.finished.connect(self._track_engine_thread.quit)
        self._track_engine_worker.finished.connect(self._track_engine_worker.deleteLater)
        self._track_engine_thread.finished.connect(self._track_engine_thread.deleteLater)

        # 5) 真正启动线程
        self._track_engine_thread.start()

    def _on_track_engine_ready(self, engine: TrackEngine):
        """后台线程加载完成时被调用（已回到主线程）"""
        self.track_engine = engine
        self.raw_view.clear_overlay()
        self.steady_view.clear_overlay()
        self.status.clearMessage()
        self._update_state(AppState.IDLE)
        self.open_btn.setEnabled(True)

    def _on_track_engine_error(self, msg: str):
        """后台初始化失败"""
        self.track_engine = None
        self.status.showMessage(f"模型加载失败: {msg}")
        self.raw_view.set_overlay(f"模型加载失败: {msg}", center=True)
        self.open_btn.setEnabled(False)

    # ---------------- UI 建立 ----------------
    def _build_ui(self):
        # 顶部工具条
        self.open_btn = QPushButton("打开视频")
        self.export_btn = QPushButton("导出视频")
        self.readme_btn = QPushButton("使用说明")
        self.feedback_btn = QPushButton("问题反馈")
        self.author_bth = QPushButton("联系我们")
        self.file_label = QLabel("未加载视频")

        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.addWidget(self.open_btn)
        top_layout.addWidget(self.export_btn)
        top_layout.addWidget(self.readme_btn)
        top_layout.addWidget(self.feedback_btn)
        top_layout.addWidget(self.author_bth)
        top_layout.addStretch(1)
        top_layout.addWidget(self.file_label)

        # 中部：左右视频 + 控制面板
        self.raw_view = VideoView("原始视频")
        self.steady_view = VideoView("稳像结果")
        self.control_panel = ControlPanel()

        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.addWidget(self.raw_view)
        center_splitter.addWidget(self.steady_view)
        center_splitter.addWidget(self.control_panel)
        center_splitter.setSizes([600, 600, 320])

        # 底部：播放控制 + 时间轴
        self.prev_btn = QToolButton()
        self.prev_btn.setText("◀")
        self.play_btn = QToolButton()
        self.play_btn.setText("▶")

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.time_label = QLabel("00:00 / 00:00")

        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.addWidget(self.prev_btn)
        bottom_layout.addWidget(self.play_btn)
        bottom_layout.addWidget(self.timeline_slider, 1)
        bottom_layout.addWidget(self.time_label)

        # 中央布局
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(top_bar)
        main_layout.addWidget(center_splitter, 1)
        main_layout.addWidget(bottom_bar)

        self.setCentralWidget(central)

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # 信号连接
        self.open_btn.clicked.connect(self._on_open_clicked)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.play_btn.clicked.connect(self._on_play_pause_clicked)
        self.prev_btn.clicked.connect(self._on_prev_clicked)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_released)

        self.control_panel.smoothChanged.connect(self._on_smooth_changed)
        self.control_panel.cropChanged.connect(self._on_crop_changed)
        self.control_panel.recompute.clicked.connect(self._on_recompute_clicked)

        self.raw_view.clicked.connect(self._on_raw_view_clicked)

    # ---------------- 一些重置工具 ----------------
    def _reset_preview_player(self):
        """停止预览计时器，释放预览用的两个 VideoCapture，并清空计数。"""
        self.preview_timer.stop()
        if self.preview_cap_raw is not None:
            self.preview_cap_raw.release()
            self.preview_cap_raw = None
        if self.preview_cap_steady is not None:
            self.preview_cap_steady.release()
            self.preview_cap_steady = None
        self.preview_total_frames = 0
        self.preview_fps = 0.0
        self.preview_frame_idx = 0

    def _reset_video_views(self):
        """把两个视频窗口的图像和 overlay 全部清空，时间轴和时间显示也重置。"""
        self.raw_view.set_frame_pixmap(None)
        self.steady_view.set_frame_pixmap(None)
        self.raw_view.clear_overlay()
        self.steady_view.clear_overlay()

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(0, 0, 1.0)  # 00:00 / 00:00

    # ---------------- 状态切换 ----------------
    def _update_state(self, new_state: AppState):
        self.state = new_state

        self.open_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.timeline_slider.setEnabled(False)
        # overlay 不一律清掉，交给具体状态控制
        # self.raw_view.clear_overlay()
        # self.steady_view.clear_overlay()

        if new_state == AppState.IDLE:
            self._reset_video_views()
            help_text = "打开视频 -> 自动跟踪 -> 规划运镜 -> 预览对比 -> 导出视频"
            self.status.showMessage(help_text)
            self.raw_view.set_overlay(help_text, center=True)

        elif new_state == AppState.TRACKING:
            self.open_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.control_panel.track_class_combo.setEnabled(False)
            self.control_panel.recompute.setEnabled(False)

            self.raw_view.set_overlay("跟踪中...\n跟踪完成后将自动规划运镜", center=False)
            self.steady_view.clear_overlay()
            self.status.showMessage("跟踪中...")

        elif new_state == AppState.TRACK_DONE:
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.status.showMessage("跟踪完成")

        elif new_state == AppState.PLANNING:
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.control_panel.recompute.setEnabled(False)

            self.raw_view.clear_overlay()
            self.steady_view.set_overlay("正在规划最优运镜路径，请稍等...", center=True)
            self.status.showMessage("正在规划最优运镜路径，请稍等...")

        elif new_state == AppState.PLAN_DONE:
            self.export_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("▶")
            self.control_panel.recompute.setEnabled(True)

            self.raw_view.set_overlay("稳像完成，可播放预览效果\n空格或下方播放键可控制播放/暂停", center=False, color=QColor(255, 200, 200))
            # self.steady_view.set_overlay("稳像完成，可播放预览效果\n空格或下方播放键可控制播放/暂停", center=False, color=QColor(255, 200, 200))
            self.status.showMessage("稳像完成，可播放预览效果")

        elif new_state == AppState.EXPORTING:
            self.open_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.steady_view.set_overlay("正在导出稳像视频...", center=True, color=QColor(220, 220, 255))
            self.status.showMessage("正在导出稳像视频...")

    # ---------------- 控件回调 ----------------
    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return

        print("video path", path)

        # 清理上一次的所有状态
        self._reset_preview_player()
        self._reset_video_views()
        self.track_play_timer.stop()
        self.track_result_all = None
        self.crop_traj = None

        if not self.track_engine:
            self.status.showMessage("算法尚未初始化完成，稍后重试")
            return

        # 关闭旧视频
        self.track_engine.finished()
        # 打开新视频
        self.track_engine.open_video(path)
        self.file_label.setText(path)

        total_frames = self.track_engine.total_frames
        self.timeline_slider.setRange(0, max(0, total_frames - 1))
        self.timeline_slider.setValue(0)
        self._update_time_label(0, total_frames, self.track_engine.fps)

        # 预览用路径
        self.preview_raw_path = self.track_engine.tmp_video_path
        self.preview_steady_path = os.path.join(utils.get_airsteady_cache_dir(), "tmp_crop.mp4")

        # Start to track.
        self.track_play_timer.start()
        self._update_state(AppState.TRACKING)

    def _on_export_clicked(self):
        # TODO: 这里未来可以做“高分辨率正式导出”
        print("Export button clicked (正式导出逻辑待实现)")

    def _on_play_pause_clicked(self):
        if self.state == AppState.PLAN_DONE and self.preview_cap_raw and self.preview_cap_raw.isOpened():
            self._toggle_preview_play_pause()

    def _toggle_preview_play_pause(self):
        if self.preview_timer.isActive():
            self.preview_timer.stop()
            self.play_btn.setText("▶")
        else:
            if self.preview_fps <= 1e-3:
                interval = 33
            else:
                interval = int(1000.0 / self.preview_fps)
            self.preview_timer.setInterval(max(1, interval))
            self.preview_timer.start()
            self.play_btn.setText("⏸")

    def _on_prev_clicked(self):
        if self.state != AppState.PLAN_DONE:
            return
        if self.preview_total_frames <= 0:
            return
        new_idx = max(0, self.preview_frame_idx - 1)
        self._seek_preview_to(new_idx)

    def _on_timeline_released(self):
        if self.state != AppState.PLAN_DONE:
            return
        if self.preview_total_frames <= 0:
            return
        frame_idx = self.timeline_slider.value()
        self._seek_preview_to(frame_idx)

    def _on_smooth_changed(self, alpha: float):
        # 镜头稳定程度 slider 改变时，先不立即重算，等用户点击“重新运镜”
        print(f"[UI] 镜头稳定程度更新: {alpha:.2f}")

    def _on_crop_changed(self, keep_ratio: float):
        # 裁切保留比例 slider 改变时，先不立即重算
        print(f"[UI] 裁切保留比例更新: {keep_ratio:.2f}")

    def _on_recompute_clicked(self):
        """PLAN_DONE 状态下，使用当前 slider 参数重新规划 + 重新导出 + 回到预览。"""
        if self.state != AppState.PLAN_DONE:
            return
        if not self.track_engine or not self.track_result_all:
            return

        print("[Recompute] 重新运镜")
        self.state = AppState.TRACK_DONE
        self.raw_view.clear_overlay()
        self.steady_view.clear_overlay()

        # 读当前 UI 参数
        smooth_ui = self.control_panel.smooth_slider.value() / 100.0
        keep_ratio = self.control_panel.crop_slider.value() / 100.0

        # 映射到算法参数
        smooth_factor = float(np.clip(1.0 - smooth_ui, 0.0, 1.0))
        # keep_ratio = 1.0 -> max_crop_ratio = 0.0 (不裁切)
        max_crop_ratio = float(np.clip(1.0 - keep_ratio, 0.0, 0.6))

        print(f"[Recompute] smooth_factor={smooth_factor:.3f}, keep_ratio={keep_ratio:.3f}, max_crop_ratio={max_crop_ratio:.3f}")

        self._update_state(AppState.PLANNING)

        # 重新规划轨迹
        self.crop_traj = algorithm.planning_crop_traj(
            track_result=self.track_result_all,
            img_width=self.track_engine.scale_width,
            img_height=self.track_engine.scale_height,
            max_crop_ratio=max_crop_ratio,
            smooth_factor=smooth_factor,
            debug=False,
        )

        # 重新导出预览小视频
        algorithm.export_stabilized_video(
            input_video_path=self.preview_raw_path,
            crop_frames=self.crop_traj,
            output_video_path=self.preview_steady_path,
            work_width=self.track_engine.scale_width,
            work_height=self.track_engine.scale_height,
            progress_cb=self._on_export_progress,
        )

        # 重新打开预览播放器
        self._init_preview_player()
        self._update_state(AppState.PLAN_DONE)

    def _on_raw_view_clicked(self, x_norm: float, y_norm: float):
        print("_on_raw_view_clicked", x_norm, y_norm)
        # 当前版本没有“点选目标”的交互，这里留作未来扩展
        self.last_click_norm = (x_norm, y_norm)

    def _on_export_progress(self, percent: float):
        # 更新状态栏 或者 overlay
        self.status.showMessage(f"正在导出稳像预览视频... {percent:.1f}%")
        self.steady_view.set_overlay(f"正在导出稳像预览视频... {percent:.1f}%", center=False)

    # ---------------- 跟踪阶段：逐帧调用 TrackEngine ----------------
    def _on_track_obj(self):
        if not self.track_engine:
            return

        vis, track_result = self.track_engine.next_frame()

        # === 1) 跟踪结束 ===
        if vis is None:
            self.track_play_timer.stop()
            self.track_engine.finished()
            self._update_state(AppState.TRACK_DONE)

            # 所有帧的跟踪结果（list[TrackFrame]）
            self.track_result_all = self.track_engine.track_results

            # 读当前 UI 参数
            smooth_ui = self.control_panel.smooth_slider.value() / 100.0
            keep_ratio = self.control_panel.crop_slider.value() / 100.0

            # 映射到算法参数
            smooth_factor = float(np.clip(smooth_ui, 0.0, 1.0))
            max_crop_ratio = float(np.clip(1.0 - keep_ratio, 0.0, 0.6))

            print(f"smooth_factor = {smooth_factor:.3f} keep_ratio = {keep_ratio:.3f} max_crop_ratio = {max_crop_ratio:.3f}")

            # 进入 PLANNING 状态（UI 显示“正在规划运镜...”）
            self._update_state(AppState.PLANNING)

            # 这里先直接在主线程里算（后面你要的话可以放到 QThread 里）
            self.crop_traj = algorithm.planning_crop_traj(
                track_result=self.track_result_all,
                img_width=self.track_engine.scale_width,
                img_height=self.track_engine.scale_height,
                max_crop_ratio=max_crop_ratio,
                smooth_factor=smooth_factor,
                debug=False,   # 方便你调试轨迹
            )

            # 根据裁切结果，生成预览用的小视频
            self.preview_raw_path = self.track_engine.tmp_video_path
            self.preview_steady_path = os.path.join(utils.get_airsteady_cache_dir(), "tmp_crop.mp4")

            algorithm.export_stabilized_video(
                input_video_path=self.preview_raw_path,
                crop_frames=self.crop_traj,
                output_video_path=self.preview_steady_path,
                work_width=self.track_engine.scale_width,
                work_height=self.track_engine.scale_height,
                progress_cb=self._on_export_progress,  # 更新状态栏/overlay
            )

            # 初始化预览播放器
            self._init_preview_player()
            # 规划 + 导出完成
            self._update_state(AppState.PLAN_DONE)
            return

        # === 2) 跟踪过程 ===
        # Show image
        self.raw_view.set_frame_pixmap(bgr_to_qpixmap(vis))

        frame_idx = track_result.frame_idx
        total_frames = self.track_engine.total_frames
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(frame_idx, total_frames, self.track_engine.fps)

        # complete ratio
        complete_ratio = 100 * float(frame_idx) / max(1, total_frames)
        self.raw_view.set_overlay(
            f"跟踪中...{int(round(complete_ratio))}%\n跟踪完成后将自动规划运镜",
            center=False,
        )

    # ---------------- 预览阶段：加载 tmp_track.mp4 & tmp_crop.mp4 ----------------
    def _init_preview_player(self):
        """在跟踪 + 规划 + 导出完成后，打开两个小视频用于预览。"""
        self._reset_preview_player()

        if not os.path.exists(self.preview_raw_path) or not os.path.exists(self.preview_steady_path):
            self.status.showMessage("预览视频文件不存在，无法预览")
            return

        self.preview_cap_raw = cv2.VideoCapture(self.preview_raw_path)
        self.preview_cap_steady = cv2.VideoCapture(self.preview_steady_path)

        if (not self.preview_cap_raw.isOpened()) or (not self.preview_cap_steady.isOpened()):
            self.status.showMessage("打开预览视频失败")
            return

        fps = self.preview_cap_raw.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = self.track_engine.fps if self.track_engine else 25.0
        self.preview_fps = float(fps)

        total_frames = int(self.preview_cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        self.preview_total_frames = total_frames
        self.preview_frame_idx = 0

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, max(0, total_frames - 1))
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(0, total_frames, self.preview_fps)

        # 先读一帧显示
        ret1, frame1 = self.preview_cap_raw.read()
        ret2, frame2 = self.preview_cap_steady.read()
        if ret1 and ret2:
            self.raw_view.set_frame_pixmap(bgr_to_qpixmap(frame1))
            self.steady_view.set_frame_pixmap(bgr_to_qpixmap(frame2))
            self._update_steady_overlay_with_clamp(0)
        else:
            self.status.showMessage("预览视频无内容")

        # 默认不自动播放，等用户按播放键或空格
        self.play_btn.setText("▶")

    def _update_steady_overlay_with_clamp(self, frame_idx: int):
        """根据当前帧的 clamp 标记，在稳像视频上显示/隐藏警告提示。"""
        warn_text = "超过最大裁切范围\n建议减少裁切保留比例后，点击重算运镜"

        show_warn = False
        if self.crop_traj is not None and 0 <= frame_idx < len(self.crop_traj):
            if getattr(self.crop_traj[frame_idx], "clamp", False):
                show_warn = True

        if show_warn:
            self.steady_view.set_overlay(
                warn_text,
                center=True,
                color=QColor(255, 200, 200),
            )
        else:
            self.steady_view.clear_overlay()

    def _on_preview_tick(self):
        """预览计时器回调：同步播放原始 & 稳像小视频。"""
        if not self.preview_cap_raw or not self.preview_cap_steady:
            self.preview_timer.stop()
            return

        ret1, frame1 = self.preview_cap_raw.read()
        ret2, frame2 = self.preview_cap_steady.read()
        if not ret1 or not ret2:
            # 播放结束
            self.preview_timer.stop()
            self.play_btn.setText("▶")
            return

        idx = self.preview_frame_idx
        self.preview_frame_idx += 1

        self.raw_view.set_frame_pixmap(bgr_to_qpixmap(frame1))
        self.steady_view.set_frame_pixmap(bgr_to_qpixmap(frame2))

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(idx)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(idx, self.preview_total_frames, self.preview_fps)

        self._update_steady_overlay_with_clamp(idx)

    def _seek_preview_to(self, frame_idx: int):
        """跳转预览到指定帧（用于拖动时间轴或上一帧）。"""
        if not self.preview_cap_raw or not self.preview_cap_steady:
            return
        if self.preview_total_frames <= 0:
            return

        frame_idx = max(0, min(frame_idx, self.preview_total_frames - 1))
        self.preview_frame_idx = frame_idx

        # 停止播放
        self.preview_timer.stop()
        self.play_btn.setText("▶")

        # 定位到指定帧
        self.preview_cap_raw.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.preview_cap_steady.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret1, frame1 = self.preview_cap_raw.read()
        ret2, frame2 = self.preview_cap_steady.read()
        if ret1 and ret2:
            self.raw_view.set_frame_pixmap(bgr_to_qpixmap(frame1))
            self.steady_view.set_frame_pixmap(bgr_to_qpixmap(frame2))

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(frame_idx, self.preview_total_frames, self.preview_fps)

        self._update_steady_overlay_with_clamp(frame_idx)

    def _update_time_label(self, frame_idx: int, total_frames: int, fps: float):
        fps = max(fps, 1e-3)
        sec_cur = int(frame_idx / fps)
        sec_all = int(total_frames / fps) if total_frames > 0 else 0

        cur_str = f"{sec_cur // 60:02d}:{sec_cur % 60:02d}"
        all_str = f"{sec_all // 60:02d}:{sec_all % 60:02d}"
        self.time_label.setText(f"{cur_str} / {all_str}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
