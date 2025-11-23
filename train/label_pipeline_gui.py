#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的打标 GUI 工具：
- 输入：视频文件路径（mp4 等）+ 抽帧 skip + 保存目录
- 流程：
  1) 调用 extract_and_prelabel.py 做抽帧 + 预标注
  2) 调用 labelImg 打开预标注结果，进行人工修标
  3) 关闭 LabelImg 后，调用 split_dataset.py 划分 train / val

说明：
- 界面只暴露 video + skip + 保存路径，其它参数在脚本顶部写死（可改）。
- 所有子流程的日志直接 print 到终端（CMD），方便看进度。
"""

import os
import sys
import subprocess
from pathlib import Path

from PySide6 import QtWidgets, QtCore


# ================== 固定参数配置 ==================
# 你可以按需改这几个默认参数

YOLO_ENV_NAME = "air_steady"   # 仅用于提示，不在代码里切换 env，由 bat 负责
MODEL_NAME = "yolo11x.pt"
IMGSZ = 1024
DEVICE = "0"

VAL_RATIO = 0.2
SEED = 42

# 默认数据集根目录：脚本所在目录下的 datasets/
SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DEFAULT_DATASETS_ROOT = SCRIPT_DIR / "datasets"

EXTRACT_SCRIPT = SCRIPT_DIR / "extract_and_prelabel.py"
SPLIT_SCRIPT = SCRIPT_DIR / "split_dataset.py"


class LabelPipelineWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AirSteady 打标助手")
        self.resize(700, 230)

        # ---- 控件：视频路径 ----
        self.video_edit = QtWidgets.QLineEdit()
        self.video_edit.setPlaceholderText("请选择要打标的视频文件 (*.mp4, *.avi, ...)")
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(self.on_browse_video)

        # ---- 控件：保存目录 ----
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setText(str(DEFAULT_DATASETS_ROOT))
        self.output_edit.setPlaceholderText("请选择打标数据保存的根目录（会在下面创建 <视频名> 和 <视频名>_split）")
        browse_output_btn = QtWidgets.QPushButton("浏览...")
        browse_output_btn.clicked.connect(self.on_browse_output)

        # ---- 控件：抽帧间隔 ----
        self.skip_spin = QtWidgets.QSpinBox()
        self.skip_spin.setRange(1, 1000)
        self.skip_spin.setValue(10)
        self.skip_spin.setSuffix(" 帧")

        # ---- 按钮 ----
        self.start_btn = QtWidgets.QPushButton("开始打标")
        self.start_btn.clicked.connect(self.on_start)

        self.close_btn = QtWidgets.QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)

        # 提示 label
        tip_label = QtWidgets.QLabel(
            "流程：预标注 → 自动打开 LabelImg 修标 → 关闭 LabelImg 后自动划分 train/val。\n"
            "保存目录下会自动创建：<视频名> 和 <视频名>_split 两个子目录。\n"
            "运行过程日志请在终端窗口中查看。"
        )
        tip_label.setWordWrap(True)

        # ---- 布局 ----
        form_layout = QtWidgets.QFormLayout()

        h_video = QtWidgets.QHBoxLayout()
        h_video.addWidget(self.video_edit)
        h_video.addWidget(browse_btn)
        form_layout.addRow("视频文件：", h_video)

        h_output = QtWidgets.QHBoxLayout()
        h_output.addWidget(self.output_edit)
        h_output.addWidget(browse_output_btn)
        form_layout.addRow("保存目录：", h_output)

        form_layout.addRow("抽帧间隔：", self.skip_spin)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.close_btn)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(form_layout)
        main_layout.addWidget(tip_label)
        main_layout.addStretch(1)
        main_layout.addLayout(btn_layout)

    # ------------------------------------------------
    # 事件处理
    # ------------------------------------------------
    def on_browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            str(SCRIPT_DIR),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.video_edit.setText(path)

    def on_browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择保存目录",
            str(DEFAULT_DATASETS_ROOT),
        )
        if path:
            self.output_edit.setText(path)

    def on_start(self):
        # 检查视频
        video_path_str = self.video_edit.text().strip()
        if not video_path_str:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一个视频文件。")
            return

        video_path = Path(video_path_str)
        if not video_path.is_file():
            QtWidgets.QMessageBox.critical(self, "错误", f"视频文件不存在：\n{video_path}")
            return

        # 检查保存目录
        output_root_str = self.output_edit.text().strip()
        if not output_root_str:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择保存目录。")
            return

        datasets_root = Path(output_root_str)
        # 这里不强制必须存在，没有则后面会自动创建
        # 但做个简单检查
        try:
            datasets_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"无法创建/访问保存目录：\n{datasets_root}\n\n错误：{e}",
            )
            return

        skip = int(self.skip_spin.value())

        # 输出目录：<保存目录>/<视频名>
        basename = video_path.stem
        output_root = datasets_root / basename
        img_dir = output_root / "images"
        label_dir = output_root / "labels"
        classes_file = label_dir / "classes.txt"
        split_output = datasets_root / f"{basename}_split"

        msg = (
            f"确认开始打标？\n\n"
            f"视频：{video_path}\n"
            f"抽帧间隔：{skip} 帧\n\n"
            f"保存根目录：{datasets_root}\n"
            f"预标注输出：{output_root}\n"
            f"划分后输出：{split_output}\n\n"
            f"提示：过程日志会打印在终端窗口中。"
        )
        ret = QtWidgets.QMessageBox.question(
            self,
            "确认开始",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        # 禁用按钮防止重复点击
        self.start_btn.setEnabled(False)
        try:
            self.run_pipeline(
                video_path=video_path,
                skip=skip,
                output_root=output_root,
                img_dir=img_dir,
                label_dir=label_dir,
                classes_file=classes_file,
                split_output=split_output,
            )
        finally:
            self.start_btn.setEnabled(True)

    # ------------------------------------------------
    # 主流程：预标注 -> LabelImg -> split_dataset
    # ------------------------------------------------
    def run_pipeline(
        self,
        video_path: Path,
        skip: int,
        output_root: Path,
        img_dir: Path,
        label_dir: Path,
        classes_file: Path,
        split_output: Path,
    ):
        # 确保上层目录存在（保存根目录已经在 on_start 里创建，这里再确保输出子目录父目录存在）
        try:
            output_root.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"无法创建输出目录：\n{output_root.parent}\n\n错误：{e}",
            )
            return

        # Step 1: 预标注
        self.log(f"[STEP 1] 预标注开始，调用 extract_and_prelabel.py ...")
        if not EXTRACT_SCRIPT.is_file():
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"找不到预标注脚本：\n{EXTRACT_SCRIPT}\n\n"
                f"请确认脚本与本 GUI 在同一目录下。",
            )
            return

        cmd1 = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "-v",
            str(video_path),
            "-o",
            str(output_root),
            "-m",
            MODEL_NAME,
            "-s",
            str(skip),
            "--imgsz",
            str(IMGSZ),
            "--device",
            DEVICE,
        ]
        self.log(f"[CMD] {' '.join(cmd1)}")
        ret1 = subprocess.run(cmd1, cwd=str(SCRIPT_DIR))
        if ret1.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"预标注脚本执行失败。\n返回码：{ret1.returncode}",
            )
            return

        if not img_dir.is_dir():
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"预标注后未找到 images 目录：\n{img_dir}",
            )
            return

        if not label_dir.is_dir():
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"预标注后未找到 labels 目录：\n{label_dir}",
            )
            return

        if not classes_file.is_file():
            self.log(f"[WARN] 未找到 classes.txt：{classes_file}")

        self.log(f"[STEP 1 DONE] 预标注完成，数据集目录：{output_root}")

        # Step 2: 打开 LabelImg
        self.log(f"[STEP 2] 打开 LabelImg 进行人工修标 ...")
        # 假设当前 conda 环境已安装 labelImg，命令可直接用
        cmd2 = ["labelImg", str(img_dir)]
        # 有 classes.txt 和 label_dir 的情况下可以都传
        if classes_file.is_file():
            cmd2.append(str(classes_file))
        cmd2.append(str(label_dir))

        self.log(f"[CMD] {' '.join(cmd2)}")
        # 这里是阻塞调用：LabelImg 关闭后才继续
        ret2 = subprocess.run(cmd2, cwd=str(SCRIPT_DIR))
        if ret2.returncode != 0:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                f"LabelImg 退出时返回码为 {ret2.returncode}。\n"
                f"如果只是正常关闭，可以忽略；如有异常，请检查。",
            )

        self.log(f"[STEP 2 DONE] LabelImg 已关闭，开始划分 train/val ...")

        # Step 3: 划分 train / val
        if not SPLIT_SCRIPT.is_file():
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"找不到划分脚本：\n{SPLIT_SCRIPT}\n\n"
                f"请确认 split_dataset.py 与本 GUI 在同一目录下。",
            )
            return

        cmd3 = [
            sys.executable,
            str(SPLIT_SCRIPT),
            "--input",
            str(output_root),
            "--output",
            str(split_output),
            "--val-ratio",
            str(VAL_RATIO),
            "--seed",
            str(SEED),
        ]
        self.log(f"[CMD] {' '.join(cmd3)}")
        ret3 = subprocess.run(cmd3, cwd=str(SCRIPT_DIR))
        if ret3.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"划分 train/val 失败。\n返回码：{ret3.returncode}",
            )
            return

        self.log(f"[STEP 3 DONE] 划分完成。")
        QtWidgets.QMessageBox.information(
            self,
            "完成",
            f"全部流程完成。\n\n"
            f"原始标注数据：\n{output_root}\n\n"
            f"train/val 划分数据：\n{split_output}",
        )

    @staticmethod
    def log(msg: str):
        # 打印到终端
        print(msg)
        sys.stdout.flush()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LabelPipelineWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
