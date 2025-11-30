import os
import sys
from typing import Optional

from PySide6.QtCore import QProcess, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QLineEdit,
)


class LabelImgLauncher(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("LabelImg Dataset Launcher")
        self.resize(600, 160)

        self.process: Optional[QProcess] = None
        self.last_data_folder: Optional[str] = None

        # 当前选中的数据目录及子路径
        self.data_folder: Optional[str] = None
        self.images_dir: Optional[str] = None
        self.labels_dir: Optional[str] = None
        self.classes_file: Optional[str] = None

        # ---------- UI ----------
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 上面一行：标签 + 文本框 + 浏览按钮
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        self.path_label = QLabel("数据目录(&D)：", self)

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText(
            "请选择 data_folder（内部包含 images / labels / classes.txt）"
        )
        self.path_edit.setReadOnly(True)

        self.browse_button = QPushButton("浏览...", self)
        self.browse_button.clicked.connect(self.on_browse_clicked)

        row_layout.addWidget(self.path_label)
        row_layout.addWidget(self.path_edit, stretch=1)
        row_layout.addWidget(self.browse_button)

        # 提示文字
        self.info_label = QLabel(
            "说明：\n"
            "  1. 点击右侧“浏览...”选择数据根目录 data_folder\n"
            "  2. 再点击下面“开始打标”启动 LabelImg\n"
            "  启动命令：labelImg data_folder/images "
            "data_folder/labels/classes.txt data_folder/labels",
            self,
        )
        self.info_label.setWordWrap(True)

        # 开始按钮
        self.start_button = QPushButton("开始打标", self)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.on_start_clicked)

        main_layout.addLayout(row_layout)
        main_layout.addWidget(self.info_label)
        main_layout.addStretch()
        main_layout.addWidget(self.start_button)

    # -------------------- 选择目录（浏览按钮） --------------------
    @Slot()
    def on_browse_clicked(self):
        """
        点击“浏览...”选择 data_folder，只做选择和检查，不立刻启动 LabelImg。
        """
        if self.process is not None:
            QMessageBox.warning(
                self,
                "正在标注中",
                "已有一个 LabelImg 进程在运行，请先关闭它再重新选择。",
            )
            return

        start_dir = self.last_data_folder or os.path.expanduser("~")

        # 不用静态的 getExistingDirectory，而是自己 new 一个对话框
        dlg = QFileDialog(self, "选择数据目录 (data_folder)")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        # 关闭原生对话框，使用 Qt 自己的，这样尺寸可以控制
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setDirectory(start_dir)

        # 手动设置一个合适的大小（按需调整）
        dlg.resize(800, 500)

        if dlg.exec() != QFileDialog.Accepted:
            return  # 用户取消

        data_folder_list = dlg.selectedFiles()
        if not data_folder_list:
            return

        data_folder = os.path.normpath(data_folder_list[0])
        self.last_data_folder = data_folder

        images_dir = os.path.join(data_folder, "images")
        labels_dir = os.path.join(data_folder, "labels")
        classes_file = os.path.join(labels_dir, "classes.txt")

        # 检查目录/文件是否存在
        missing = []
        if not os.path.isdir(images_dir):
            missing.append(f"缺少目录: {images_dir}")
        if not os.path.isdir(labels_dir):
            missing.append(f"缺少目录: {labels_dir}")
        if not os.path.isfile(classes_file):
            missing.append(f"缺少文件: {classes_file}")

        if missing:
            QMessageBox.critical(
                self,
                "数据目录结构不正确",
                "以下目录/文件不存在，请检查数据结构：\n\n"
                + "\n".join(missing),
            )
            self.data_folder = None
            self.images_dir = None
            self.labels_dir = None
            self.classes_file = None
            self.path_edit.setText("")
            self.start_button.setEnabled(False)
            return

        # 记录当前选择
        self.data_folder = data_folder
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file

        self.path_edit.setText(data_folder)
        self.start_button.setEnabled(True)

    # -------------------- 开始打标 --------------------
    @Slot()
    def on_start_clicked(self):
        """
        在已经选择好 data_folder 后，点击“开始打标”才真正启动 LabelImg。
        """
        if self.process is not None:
            QMessageBox.warning(
                self,
                "正在标注中",
                "已有一个 LabelImg 进程在运行，请先关闭它再开始下一次打标。",
            )
            return

        if not self.data_folder or not self.images_dir or not self.labels_dir or not self.classes_file:
            QMessageBox.warning(
                self,
                "未选择数据目录",
                "请先点击“浏览...”选择数据目录 (data_folder) 再开始打标。",
            )
            return

        # 再次简单检查一下
        if not (os.path.isdir(self.images_dir)
                and os.path.isdir(self.labels_dir)
                and os.path.isfile(self.classes_file)):
            QMessageBox.critical(
                self,
                "数据目录已失效",
                "当前数据目录结构已发生变化，请重新选择数据目录。",
            )
            self.start_button.setEnabled(False)
            return

        self.launch_labelimg(self.images_dir, self.classes_file, self.labels_dir)

    def launch_labelimg(self, images_dir: str, classes_file: str, labels_dir: str):
        """
        使用 QProcess 启动 labelImg，不阻塞 UI。
        """
        # 方案1：命令行能直接执行 labelImg
        program = "labelImg"
        args = [images_dir, classes_file, labels_dir]

        # 如果只能用 python -m labelImg，则改为：
        # program = sys.executable
        # args = ["-m", "labelImg", images_dir, classes_file, labels_dir]

        self.process = QProcess(self)
        self.process.setProgram(program)
        self.process.setArguments(args)

        self.process.finished.connect(self.on_labelimg_finished)
        self.process.errorOccurred.connect(self.on_labelimg_error)

        self.browse_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.start_button.setText("LabelImg 运行中，请完成标注后关闭...")

        self.process.start()

        if not self.process.waitForStarted(3000):
            # 启动失败
            self.process = None
            self.browse_button.setEnabled(True)
            self.start_button.setEnabled(self.data_folder is not None)
            self.start_button.setText("开始打标")

            QMessageBox.critical(
                self,
                "启动失败",
                "无法启动 labelImg。\n"
                "请确认：\n"
                "1. 已安装 labelImg（pip install labelImg）\n"
                "2. 在终端中可以直接运行 'labelImg'\n"
                "如不行，可将脚本中的 program 改为：python -m labelImg",
            )

    @Slot(int, QProcess.ExitStatus)
    def on_labelimg_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """
        labelImg 关闭后的回调，可以重新选择或继续用当前目录打标。
        """
        self.process = None
        self.browse_button.setEnabled(True)
        self.start_button.setEnabled(self.data_folder is not None)
        self.start_button.setText("开始打标")

        if exit_status == QProcess.NormalExit:
            msg = f"LabelImg 已退出，退出码：{exit_code}\n可以重新选择新的数据目录，或继续用当前目录打标。"
        else:
            msg = "LabelImg 异常退出，可以尝试重新运行或检查命令。"

        QMessageBox.information(self, "LabelImg 已关闭", msg)

    @Slot(QProcess.ProcessError)
    def on_labelimg_error(self, error: QProcess.ProcessError):
        """
        启动 / 运行过程中出现错误。
        """
        self.process = None
        self.browse_button.setEnabled(True)
        self.start_button.setEnabled(self.data_folder is not None)
        self.start_button.setText("开始打标")

        QMessageBox.critical(
            self,
            "LabelImg 进程错误",
            f"LabelImg 运行出错 (error={error})，请检查安装与命令配置。",
        )


def main():
    app = QApplication(sys.argv)
    window = LabelImgLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
