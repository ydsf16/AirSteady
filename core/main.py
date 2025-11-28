import sys
from enum import Enum, auto
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal, QThread, QObject, Slot
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
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
)

from algorithm import TrackEngine
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
    smoothChanged = Signal(float)     # alpha
    cropChanged = Signal(float)       # ratio
    modeChanged = Signal(str)         # "semi" / "auto"

    def __init__(self, parent=None):
        super().__init__(parent)

        # 预设（现在放在最上面）
        track_class_label = QLabel("跟踪对象")
        self.track_class_combo = QComboBox()
        self.track_class_combo.addItems(["飞机"])

        # 平滑度
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(80)
        self.smooth_value_label = QLabel("0.80")
        smooth_label = QLabel("平滑度")

        # 裁切比例
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setRange(40, 100)  # 避免太小，默认 0.8
        self.crop_slider.setValue(80)
        self.crop_value_label = QLabel("0.80")
        crop_label = QLabel("裁切比例")

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
        layout.addWidget(self.smooth_slider)
        layout.addWidget(self.smooth_value_label)

        layout.addSpacing(10)
        layout.addWidget(crop_label)
        layout.addWidget(self.crop_slider)
        layout.addWidget(self.crop_value_label)
        layout.addWidget(self.recompute)
        layout.addStretch(1)
        self.setFixedWidth(280)

        # 信号连接
        self.smooth_slider.valueChanged.connect(self._on_smooth_changed)
        self.crop_slider.valueChanged.connect(self._on_crop_changed)

    def _on_smooth_changed(self, v: int):
        alpha = v / 100.0
        self.smooth_value_label.setText(f"{alpha:.2f}")
        self.smoothChanged.emit(alpha)

    def _on_crop_changed(self, v: int):
        ratio = v / 100.0
        self.crop_value_label.setText(f"{ratio:.2f}")
        self.cropChanged.emit(ratio)

    def get_export_size(self, orig_width: int, orig_height: int):
        """
        根据下拉框返回导出分辨率：
        - None 表示使用原始分辨率
        - 否则 (w, h)
        """
        text = self.res_combo.currentText()
        if text == "原始分辨率":
            return None
        try:
            w_str, h_str = text.split("x")
            w = int(w_str)
            h = int(h_str)
            # 简单防御：如果原始分辨率比目标小很多，可以根据需要缩放，这里直接返回目标
            return (w, h)
        except Exception:
            return (orig_width, orig_height)


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

        # 记录用户最后一次点击的归一化坐标，用于导出时复用
        self.last_click_norm = None  # (x_norm, y_norm) 或 None

        self.track_play_timer = QTimer(self)
        self.track_play_timer.setInterval(2) # No delay 2ms
        self.track_play_timer.timeout.connect(self._on_track_obj)

        self._build_ui()
        self._update_state(AppState.IDLE)

        # Load Model & init track engine.
        # self.track_engine = TrackEngine()
        self.track_engine: TrackEngine | None = None
        self._track_engine_thread: QThread | None = None
        self._start_track_engine_init()
    
    def _start_track_engine_init(self):
        """在后台线程中初始化 TrackEngine（加载 YOLO 模型）"""
        # 提示一下用户
        self.status.showMessage("算法加载中，约需几秒，请稍后...")
        self.raw_view.set_overlay("算法加载中，约需几秒，请稍后...", center=True)
        # self.steady_view.set_overlay("算法加载中，请稍后...", center=True)
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
        # 如果你想，加载完后也可以把按钮文字/颜色改一下

    def _on_track_engine_error(self, msg: str):
        """后台初始化失败"""
        self.track_engine = None
        self.status.showMessage(f"模型加载失败: {msg}")
        self.raw_view.set_overlay(f"模型加载失败: {msg}", center=True)
        # self.steady_view.set_overlay(f"模型加载失败: {msg}", center=True)
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
        # top_layout.addStretch(1)
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
        # self.status.showMessage("准备就绪 - 请先打开视频 - 开始处理")

        # 信号连接
        self.open_btn.clicked.connect(self._on_open_clicked)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.play_btn.clicked.connect(self._on_play_pause_clicked)
        self.prev_btn.clicked.connect(self._on_prev_clicked)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_released)

        self.control_panel.smoothChanged.connect(self._on_smooth_changed)
        self.control_panel.cropChanged.connect(self._on_crop_changed)
        self.control_panel.modeChanged.connect(self._on_mode_changed)

        self.raw_view.clicked.connect(self._on_raw_view_clicked)

    def _on_export_progress(self, percent: float):
        # 更新状态栏 或者 overlay
        self.status.showMessage(f"正在导出稳像视频... {percent:.1f}%")
        self.steady_view.set_overlay(f"正在导出稳像视频... {percent:.1f}%", center=False)

    # ---------------- 状态切换 ----------------
    def _update_state(self, new_state: AppState):
        self.state = new_state

        self.open_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.timeline_slider.setEnabled(False)
        self.raw_view.clear_overlay()
        self.steady_view.clear_overlay()

        if new_state == AppState.IDLE:
            help_text = "打开视频 -> 等待自动稳像 -> [可选：参数调节&重新运镜] -> 导出视频"
            self.status.showMessage(help_text)
            self.raw_view.set_overlay(help_text, center=True)
            # self.steady_view.set_overlay(help_text, center=True)

        elif new_state == AppState.TRACKING:
            self.open_btn.setEnabled(True)
            self.export_btn.setEnabled(False)        
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.control_panel.track_class_combo.setEnabled(False)
            self.control_panel.recompute.setEnabled(False)

            self.raw_view.set_overlay("跟踪中... \n跟踪完成后将进行自动运镜", center=False)
            self.steady_view.clear_overlay()
            self.status.showMessage("跟踪中...")

        elif new_state == AppState.TRACK_DONE:
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            # self.play_btn.setText("▶")
            self.status.showMessage("跟踪完成")

        elif new_state == AppState.PLANNING:
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            # self.play_btn.setText("⏸")
            self.raw_view.clear_overlay()
            self.steady_view.set_overlay("正在规划最优运镜路径，请稍等...", center=True)
            self.status.showMessage("正在规划最优运镜路径，请稍等...")

        elif new_state == AppState.PLAN_DONE:
            self.export_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("▶")
            self.raw_view.set_overlay("规划完成，请点击播放按钮预览结果", QColor(255, 200, 200))
            self.status.showMessage("规划完成，请点击播放按钮预览结果")

        elif new_state == AppState.EXPORTING:
            self.open_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.timeline_slider.setEnabled(False)
            self.steady_view.set_overlay("正在导出稳像视频...", QColor(220, 220, 255))
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

        # Reset track engin
        self.track_play_timer.stop()
        self.track_engine.finished()
        self.track_engine.open_video(path)
        self.file_label.setText(path)

        total_frames = self.track_engine.total_frames
        self.timeline_slider.setRange(0, max(0, total_frames - 1))

        # Start to track.
        self.track_play_timer.start()
        self._update_state(AppState.TRACKING)

    def _on_export_clicked(self):
        print("waitting")

    def _on_play_pause_clicked(self):
        print("pause")

    def _on_prev_clicked(self):
        print("_on_prev_clicked")

    def _on_timeline_released(self):
        print("_on_timeline_released")

    def _on_smooth_changed(self, alpha: float):
        print("_on_smooth_changed")
        # if self.engine:
        #     self.engine.ema_alpha = float(alpha)

    def _on_crop_changed(self, ratio: float):
        print("_on_smooth_changed")

    def _on_raw_view_clicked(self, x_norm: float, y_norm: float):
        print("_on_raw_view_clicked")

    # ---------------- 播放控制 ----------------
    def _start_playback(self):
        # 预览状态可用空格触发
        print("x")

    def _pause_playback(self):
        # 预览状态 可用空格触发
        print("x")

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
            track_result_all = self.track_engine.track_results

            # 读当前 UI 参数
            smooth_factor = self.control_panel.smooth_slider.value() / 100.0
            max_crop_ratio = self.control_panel.crop_slider.value() / 100.0
            print(smooth_factor, max_crop_ratio)

            # 进入 PLANNING 状态（UI 显示“正在规划运镜...”）
            self._update_state(AppState.PLANNING)

            # 这里先直接在主线程里算（后面你要的话可以放到 QThread 里）
            self.crop_traj = algorithm.planning_crop_traj(
                track_result=track_result_all,
                img_width=self.track_engine.scale_width,
                img_height=self.track_engine.scale_height,
                max_crop_ratio=max_crop_ratio,
                smooth_factor=smooth_factor,
                debug=False,   # 方便你调试轨迹
            )

            # 规划完成
            self._update_state(AppState.PLAN_DONE)

            # TODO：这里开始根据裁切结果：self.crop_traj，进行
            algorithm.export_stabilized_video(
                input_video_path=self.track_engine.tmp_video_path,
                crop_frames=self.crop_traj,
                output_video_path=os.path.join(utils.get_airsteady_cache_dir(), "tmp_crop.mp4"),
                work_width=self.track_engine.scale_width,
                work_height=self.track_engine.scale_height,
                progress_cb=self._on_export_progress,  # 比如更新状态栏/overlay
            )

            # TODO: 这里后面可以加一行，按当前时间轴选一帧稳像结果，刷新 steady_view
            

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
        complete_ratio = 100 * float(frame_idx) / total_frames
        self.raw_view.set_overlay(
            f"跟踪中...{int(round(complete_ratio))}%\n跟踪完成后将进行自动运镜",
            center=False,
        )


    def _seek_to_frame(self, frame_idx: int):
        print("_seek_to_frame")

    def _update_time_label(self, frame_idx: int, total_frames: int, fps: float):
        fps = max(fps, 1e-3)
        sec_cur = int(frame_idx / fps)
        sec_all = int(total_frames / fps)

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
