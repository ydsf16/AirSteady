import sys
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
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

from algorithm import AirSteadyEngine


class AppState(Enum):
    IDLE = auto()                # 没有加载视频
    LOADED_NO_TARGET = auto()    # 已加载视频，但未选目标
    READY = auto()               # 已选目标，可以播放
    PLAYING = auto()             # 正在播放+跟踪
    LOST = auto()                # 目标丢失，等待用户重新点击
    EXPORTING = auto()           # 正在导出视频


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

        self.setMinimumSize(320, 240)

    def set_frame_pixmap(self, pix: QPixmap | None):
        self._pixmap = pix
        self.update()

    def set_overlay(self, text: str | None, color: QColor | None = None):
        self._overlay_text = text
        if color is not None:
            self._overlay_color = color
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
            ty = content_rect.center().y() - text_height / 2

            bg_rect = QRectF(
                tx - 10,
                ty - text_height,
                text_width + 20,
                text_height + 10,
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
        preset_label = QLabel("预设")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["默认", "更稳", "更跟随"])

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

        # 跟丢策略
        mode_label = QLabel("跟丢策略")
        self.radio_semi = QRadioButton("半自动（丢失时暂停等待你点）")
        self.radio_auto = QRadioButton("全自动（尝试自己接上）")
        self.radio_semi.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_semi)
        self.mode_group.addButton(self.radio_auto)

        # 点击目标后自动播放
        self.auto_start_checkbox = QCheckBox("点击目标后自动开始播放")
        self.auto_start_checkbox.setChecked(True)

        # 导出分辨率
        res_label = QLabel("导出分辨率")
        self.res_combo = QComboBox()
        self.res_combo.addItems([
            "原始分辨率",
            "1920x1080",
            "1280x720",
            "854x480",
        ])

        # 高级按钮（暂时占位）
        self.advanced_btn = QPushButton("高级...")

        # 布局
        layout = QVBoxLayout(self)
        title = QLabel("控制面板")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(10)
        # 预设在最顶部
        layout.addWidget(preset_label)
        layout.addWidget(self.preset_combo)

        layout.addSpacing(10)
        layout.addWidget(smooth_label)
        layout.addWidget(self.smooth_slider)
        layout.addWidget(self.smooth_value_label)

        layout.addSpacing(10)
        layout.addWidget(crop_label)
        layout.addWidget(self.crop_slider)
        layout.addWidget(self.crop_value_label)

        layout.addSpacing(15)
        layout.addWidget(mode_label)
        layout.addWidget(self.radio_semi)
        layout.addWidget(self.radio_auto)

        layout.addSpacing(10)
        layout.addWidget(self.auto_start_checkbox)

        layout.addSpacing(15)
        layout.addWidget(res_label)
        layout.addWidget(self.res_combo)

        layout.addStretch(1)
        layout.addWidget(self.advanced_btn)

        self.setFixedWidth(320)

        # 信号连接
        self.smooth_slider.valueChanged.connect(self._on_smooth_changed)
        self.crop_slider.valueChanged.connect(self._on_crop_changed)
        self.radio_semi.toggled.connect(self._on_mode_toggled)
        # 预设逻辑目前先不改参数，你之后可以按需求填（比如切换平滑度、裁切比例）

    def _on_smooth_changed(self, v: int):
        alpha = v / 100.0
        self.smooth_value_label.setText(f"{alpha:.2f}")
        self.smoothChanged.emit(alpha)

    def _on_crop_changed(self, v: int):
        ratio = v / 100.0
        self.crop_value_label.setText(f"{ratio:.2f}")
        self.cropChanged.emit(ratio)

    def _on_mode_toggled(self, checked: bool):
        if not checked:
            return
        mode = "semi" if self.radio_semi.isChecked() else "auto"
        self.modeChanged.emit(mode)

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AirSteady 飞影稳拍 - 内测版")
        self.resize(1400, 800)

        self.state = AppState.IDLE
        self.engine: AirSteadyEngine | None = None

        # 记录用户最后一次点击的归一化坐标，用于导出时复用
        self.last_click_norm = None  # (x_norm, y_norm) 或 None

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(40)  # 默认约 25fps，后面按实际 fps 调整
        self.play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._update_state(AppState.IDLE)

    # ---------------- UI 建立 ----------------
    def _build_ui(self):
        # 顶部工具条
        self.open_btn = QPushButton("打开视频")
        self.export_btn = QPushButton("导出视频")
        self.file_label = QLabel("未加载视频")

        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.addWidget(self.open_btn)
        top_layout.addWidget(self.export_btn)
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
        self.status.showMessage("准备就绪 - 请先打开视频")

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
            self.status.showMessage("准备就绪 - 请先打开视频")

        elif new_state == AppState.LOADED_NO_TARGET:
            self.export_btn.setEnabled(False)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(False)
            self.raw_view.set_overlay("请点击你要跟踪的目标")
            self.status.showMessage("已加载视频，请在左侧画面点击你要跟踪的目标")

        elif new_state == AppState.READY:
            self.export_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("▶")
            self.status.showMessage("已选中目标，点击播放开始稳像")

        elif new_state == AppState.PLAYING:
            self.export_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(False)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("⏸")
            self.status.showMessage("稳定跟踪中... 空格或点击暂停按钮可暂停")

        elif new_state == AppState.LOST:
            self.export_btn.setEnabled(True)
            self.timeline_slider.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_btn.setText("▶")
            self.raw_view.set_overlay("目标丢失，请在当前画面重新点击目标", QColor(255, 200, 200))
            self.status.showMessage("目标丢失，请在当前画面重新点击目标")

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

        # 初始化算法引擎
        crop_ratio = self.control_panel.crop_slider.value() / 100.0
        ema_alpha = self.control_panel.smooth_slider.value() / 100.0
        mode = "semi" if self.control_panel.radio_semi.isChecked() else "auto"

        self.engine = AirSteadyEngine(
            crop_ratio=crop_ratio,
            ema_alpha=ema_alpha,
            mode=mode,
        )
        self.last_click_norm = None

        self.engine.open_video(path)
        self.file_label.setText(path)

        total_frames = self.engine.total_frames
        self.timeline_slider.setRange(0, max(0, total_frames - 1))

        # 读取第一帧（带检测）
        vis, steady, info = self.engine.next_frame()
        if vis is not None:
            self.raw_view.set_frame_pixmap(bgr_to_qpixmap(vis))
            self.steady_view.set_frame_pixmap(bgr_to_qpixmap(steady))
            self.timeline_slider.setValue(info["frame_idx"])
            self._update_time_label(info["frame_idx"], info["total_frames"], self.engine.fps)

        self._update_state(AppState.LOADED_NO_TARGET)

    def _on_export_clicked(self):
        if not self.engine or self.state in (AppState.IDLE, AppState.LOADED_NO_TARGET):
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出文件路径",
            "",
            "Video Files (*.mp4)",
        )
        if not save_path:
            return

        self._update_state(AppState.EXPORTING)

        # 导出分辨率
        export_size = self.control_panel.get_export_size(
            self.engine.width, self.engine.height
        )

        def progress_callback(frame_idx, total_frames):
            percent = int(frame_idx / max(1, total_frames) * 100)
            self.steady_view.set_overlay(
                f"正在导出稳像视频... {percent}%",
                QColor(220, 220, 255)
            )
            self.status.showMessage(f"正在导出稳像视频... {percent}%")
            QApplication.processEvents()

        try:
            self.engine.export_video(
                save_path,
                progress_callback=progress_callback,
                click_norm=self.last_click_norm,
                export_size=export_size,
            )
        except Exception as e:
            self.status.showMessage(f"导出失败: {e}")
        else:
            self.status.showMessage(f"导出完成: {save_path}")
            self.steady_view.set_overlay("导出完成，已保存文件", QColor(200, 255, 200))
        finally:
            if self.engine and self.engine.target_track_id is not None:
                self._update_state(AppState.READY)
            else:
                self._update_state(AppState.LOADED_NO_TARGET)

    def _on_play_pause_clicked(self):
        if self.state == AppState.PLAYING:
            self._pause_playback()
        elif self.state in (AppState.READY, AppState.LOST):
            self._start_playback()

    def _on_prev_clicked(self):
        # 简单实现：往前一帧（会重置跟踪和缓存）
        if not self.engine:
            return
        idx = max(0, self.engine.current_frame_idx - 1)
        self._seek_to_frame(idx)

    def _on_timeline_released(self):
        if not self.engine:
            return
        frame_idx = self.timeline_slider.value()
        self._seek_to_frame(frame_idx)

    def _on_smooth_changed(self, alpha: float):
        if self.engine:
            self.engine.ema_alpha = float(alpha)

    def _on_crop_changed(self, ratio: float):
        if self.engine:
            self.engine.crop_ratio = float(ratio)
            if self.engine.width > 0 and self.engine.height > 0:
                self.engine.crop_w = int(self.engine.width * self.engine.crop_ratio)
                self.engine.crop_h = int(self.engine.height * self.engine.crop_ratio)

    def _on_mode_changed(self, mode: str):
        if self.engine:
            self.engine.set_mode(mode)

    def _on_raw_view_clicked(self, x_norm: float, y_norm: float):
        if not self.engine:
            return
        if self.state in (AppState.IDLE, AppState.EXPORTING):
            return

        # 1) 自己记一份，导出时复用
        self.last_click_norm = (x_norm, y_norm)

        # 2) 告诉引擎（引擎会在当前帧 boxes 上找最近的那个）
        self.engine.set_pending_click(x_norm, y_norm)

        # 第一次选目标：LOADED_NO_TARGET / LOST -> READY
        if self.state in (AppState.LOADED_NO_TARGET, AppState.LOST):
            self._update_state(AppState.READY)

        # 自动开始播放
        if self.control_panel.auto_start_checkbox.isChecked():
            self._start_playback()

    # ---------------- 播放控制 ----------------
    def _start_playback(self):
        if not self.engine:
            return
        if self.state not in (AppState.READY, AppState.LOST):
            return

        interval = int(1000.0 / max(1.0, self.engine.fps))
        self.play_timer.setInterval(interval)
        self.play_timer.start()
        self._update_state(AppState.PLAYING)

    def _pause_playback(self):
        self.play_timer.stop()
        if self.engine and self.engine.target_track_id is not None:
            self._update_state(AppState.READY)
        else:
            self._update_state(AppState.LOADED_NO_TARGET)

    def _on_play_tick(self):
        if not self.engine:
            return

        vis, steady, info = self.engine.next_frame()
        if vis is None:
            self.play_timer.stop()
            if self.engine.target_track_id is not None:
                self._update_state(AppState.READY)
            else:
                self._update_state(AppState.LOADED_NO_TARGET)
            return

        self.raw_view.set_frame_pixmap(bgr_to_qpixmap(vis))
        self.steady_view.set_frame_pixmap(bgr_to_qpixmap(steady))

        frame_idx = info["frame_idx"]
        total_frames = info["total_frames"]
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_idx)
        self.timeline_slider.blockSignals(False)
        self._update_time_label(frame_idx, total_frames, self.engine.fps)

        # 半自动模式下，如果丢失，则暂停并进入 LOST 状态
        if info["is_lost"] and (self.engine.mode == "semi"):
            self.play_timer.stop()
            self._update_state(AppState.LOST)

    def _seek_to_frame(self, frame_idx: int):
        if not self.engine or not self.engine.cap:
            return

        self.play_timer.stop()

        # 跳转到指定帧，并重置 tracking + 缓存
        self.engine.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.engine.current_frame_idx = frame_idx
        self.engine.reset_tracking(clear_click=True, clear_cache=True)
        self.last_click_norm = None

        vis, steady, info = self.engine.next_frame()
        if vis is None:
            return

        self.raw_view.set_frame_pixmap(bgr_to_qpixmap(vis))
        self.steady_view.set_frame_pixmap(bgr_to_qpixmap(steady))

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(info["frame_idx"])
        self.timeline_slider.blockSignals(False)
        self._update_time_label(info["frame_idx"], info["total_frames"], self.engine.fps)

        self._update_state(AppState.LOADED_NO_TARGET)

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
