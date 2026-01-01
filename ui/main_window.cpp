#include "main_window.hpp"

#include "video_view.hpp"
#include "control_panel.hpp"
#include "export_dialog.hpp"
#include "readme_dialog.hpp"
#include "contact_dialog.hpp"

#include <QApplication>
#include <QCloseEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProgressBar>
#include <QSlider>
#include <QStatusBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <QSplitter>
#include <QUrl>
#include <QEvent>
#include <QKeyEvent>
#include <QDebug>

namespace airsteady {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("AirSteady 航迹稳拍 - Beta"));
  resize(1400, 800);

  buildUi();
  updateState(AppState::kIdle);
}

MainWindow::~MainWindow() {
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
  if (event->type() == QEvent::KeyPress) {
    auto* key_event = static_cast<QKeyEvent*>(event);
    if (key_event->key() == Qt::Key_Space) {
      if (state_ == AppState::kVideoLoaded ||
          state_ == AppState::kPlaying ||
          state_ == AppState::kPaused) {
        togglePlayPause();
      }
      return true;
    }
  }
  return QMainWindow::eventFilter(obj, event);
}

void MainWindow::closeEvent(QCloseEvent* event) {
  QMainWindow::closeEvent(event);
}

void MainWindow::buildUi() {
  // 顶部工具条
  open_btn_ = new QPushButton(tr("打开视频"), this);
  export_btn_ = new QPushButton(tr("导出视频"), this);
  readme_btn_ = new QPushButton(tr("使用说明"), this);
  feedback_btn_ = new QPushButton(tr("问题反馈"), this);
  author_btn_ = new QPushButton(tr("关于作者"), this);
  file_label_ = new QLabel(tr("未加载视频"), this);

  auto* top_bar = new QWidget(this);
  auto* top_layout = new QHBoxLayout(top_bar);
  top_layout->addWidget(open_btn_);
  top_layout->addWidget(readme_btn_);
  top_layout->addWidget(feedback_btn_);
  top_layout->addWidget(author_btn_);
  top_layout->addStretch(1);
  top_layout->addWidget(export_btn_);
  top_layout->addWidget(file_label_);

  // 中间：左右视频 + 控制面板
  raw_view_ = new VideoView(tr("原始视频"), this);
  steady_view_ = new VideoView(tr("稳像结果"), this);
  control_panel_ = new ControlPanel(this);

  auto* center_splitter = new QSplitter(Qt::Horizontal, this);
  center_splitter->addWidget(raw_view_);
  center_splitter->addWidget(steady_view_);
  center_splitter->addWidget(control_panel_);
  center_splitter->setStretchFactor(0, 1);
  center_splitter->setStretchFactor(1, 2);
  center_splitter->setStretchFactor(2, 0);
  center_splitter->setSizes({400, 800, 320});
  raw_view_->setMinimumWidth(240);
  steady_view_->setMinimumWidth(360);

  // 底部播放控制
  prev_btn_ = new QToolButton(this);
  prev_btn_->setText(QStringLiteral("◀"));
  play_btn_ = new QToolButton(this);
  play_btn_->setText(QStringLiteral("▶"));

  timeline_slider_ = new QSlider(Qt::Horizontal, this);
  timeline_slider_->setRange(0, 0);
  timeline_slider_->setValue(0);
  time_label_ = new QLabel(QStringLiteral("00:00 / 00:00"), this);

  auto* bottom_bar = new QWidget(this);
  auto* bottom_layout = new QHBoxLayout(bottom_bar);
  bottom_layout->addWidget(prev_btn_);
  bottom_layout->addWidget(play_btn_);
  bottom_layout->addWidget(timeline_slider_, 1);
  bottom_layout->addWidget(time_label_);

  auto* central = new QWidget(this);
  auto* main_layout = new QVBoxLayout(central);
  main_layout->addWidget(top_bar);
  main_layout->addWidget(center_splitter, 1);
  main_layout->addWidget(bottom_bar);
  setCentralWidget(central);

  // 状态栏
  status_bar_ = new QStatusBar(this);
  setStatusBar(status_bar_);

  export_progress_ = new QProgressBar(this);
  export_progress_->setRange(0, 100);
  export_progress_->setValue(0);
  export_progress_->setVisible(false);
  status_bar_->addPermanentWidget(export_progress_);

  // 信号连接
  connect(open_btn_, &QPushButton::clicked,
          this, &MainWindow::onOpenClicked);
  connect(export_btn_, &QPushButton::clicked,
          this, &MainWindow::onExportClicked);
  connect(readme_btn_, &QPushButton::clicked,
          this, &MainWindow::onReadmeClicked);
  connect(feedback_btn_, &QPushButton::clicked,
          this, &MainWindow::onFeedbackClicked);
  connect(author_btn_, &QPushButton::clicked,
          this, &MainWindow::onContactClicked);

  connect(play_btn_, &QToolButton::clicked,
          this, &MainWindow::onPlayPauseClicked);
  connect(prev_btn_, &QToolButton::clicked,
          this, &MainWindow::onPrevClicked);
  connect(timeline_slider_, &QSlider::sliderReleased,
          this, &MainWindow::onTimelineReleased);
  
  // 参数变化相关
  connect(control_panel_, &ControlPanel::smoothChanged,
          this, &MainWindow::onSmoothChanged);
  connect(control_panel_, &ControlPanel::cropKeepRatioChanged,
          this, &MainWindow::onCropKeeRatioChanged);
  
  // 视频点击相关的，用于停止和启动播放。
  connect(raw_view_, &VideoView::clicked,
          this, &MainWindow::onVideoViewClicked);
  connect(steady_view_, &VideoView::clicked,
          this, &MainWindow::onVideoViewClicked);
}

void MainWindow::resetPreview() {

}

void MainWindow::resetVideoViews() {

}

void MainWindow::updateState(AppState new_state) {
  state_ = new_state;

  open_btn_->setEnabled(true);
  export_btn_->setEnabled(false);
  play_btn_->setEnabled(false);
  prev_btn_->setEnabled(false);
  timeline_slider_->setEnabled(false);

  raw_view_->setPlayIconVisible(false);
  steady_view_->setPlayIconVisible(false);

  switch (new_state) {
    case AppState::kIdle: {
      resetPreview();
      resetVideoViews();
      QString help_text =
          tr("打开视频 -> 预览原始画面和稳像结果（当前稳像为占位显示）");
      status_bar_->showMessage(help_text);
      raw_view_->setOverlay(help_text, true);
      steady_view_->setOverlay(help_text, true);
      control_panel_->recomputeButton()->setEnabled(false);
      break;
    }
    case AppState::kVideoLoaded: {
      export_btn_->setEnabled(true);  // 当前导出仅做占位
      play_btn_->setEnabled(true);
      prev_btn_->setEnabled(true);
      timeline_slider_->setEnabled(true);
      control_panel_->recomputeButton()->setEnabled(false);
      status_bar_->showMessage(tr("视频已加载，可以预览。"));
      raw_view_->clearOverlay();
      steady_view_->clearOverlay();
      raw_view_->setPlayIconVisible(true);
      steady_view_->setPlayIconVisible(true);
      break;
    }
    case AppState::kPlaying: {
      export_btn_->setEnabled(true);
      play_btn_->setEnabled(true);
      prev_btn_->setEnabled(true);
      timeline_slider_->setEnabled(true);
      status_bar_->showMessage(tr("播放中"));
      raw_view_->setPlayIconVisible(false);
      steady_view_->setPlayIconVisible(false);
      break;
    }
    case AppState::kPaused: {
      export_btn_->setEnabled(true);
      play_btn_->setEnabled(true);
      prev_btn_->setEnabled(true);
      timeline_slider_->setEnabled(true);
      status_bar_->showMessage(tr("已暂停"));
      raw_view_->setPlayIconVisible(true);
      steady_view_->setPlayIconVisible(true);
      break;
    }
    case AppState::kExporting: {
      open_btn_->setEnabled(false);
      export_btn_->setEnabled(false);
      play_btn_->setEnabled(false);
      prev_btn_->setEnabled(false);
      timeline_slider_->setEnabled(false);
      control_panel_->recomputeButton()->setEnabled(false);
      status_bar_->showMessage(tr("正在导出视频（占位实现）..."));
      export_progress_->setVisible(true);
      export_progress_->setValue(0);
      steady_view_->setOverlay(
          tr("正在导出视频（占位实现）..."), true, QColor(220, 220, 255));
      break;
    }
  }
}

void MainWindow::onOpenClicked() {
  QString path = QFileDialog::getOpenFileName(
      this,
      tr("选择视频文件"),
      QString(),
      tr("Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"));

  if (path.isEmpty()) {
    return;
  }

  // Check path.

}

void MainWindow::onExportClicked() {
  ExportDialog dlg(this, "video_name.mp4");
  if (dlg.exec() != QDialog::Accepted) {
    return;
  }


}

void MainWindow::onPlayPauseClicked() {
}

void MainWindow::togglePlayPause() {
  
}

void MainWindow::onPrevClicked() {
  
}

void MainWindow::onTimelineReleased() {
  
}

void MainWindow::onReadmeClicked() {
  ReadmeDialog dlg(this);
  dlg.exec();
}

void MainWindow::onFeedbackClicked() {
  // 按你的要求：问题反馈内部逻辑先清空，这里给一个占位提示即可。
  QMessageBox::information(
      this,
      tr("问题反馈"),
      tr("当前 C++ 版本的问题反馈逻辑尚未实现，"
         "后续可以接入你现有的反馈打包流程。"));
}

void MainWindow::onContactClicked() {
  ContactDialog dlg(this);
  dlg.exec();
}

void MainWindow::onSmoothChanged(double alpha) {
  qDebug() << "[UI] 镜头稳定程度更新:" << alpha;
  // 当前 Demo 不触发重新运镜，未来可以在这里调用 Processor.UpdateParam...
}

void MainWindow::onCropKeeRatioChanged(double keep_ratio) {
  qDebug() << "[UI] 裁切保留比例更新:" << keep_ratio;
}

void MainWindow::onVideoViewClicked(double /*x_norm*/,
                                    double /*y_norm*/) {
}

void MainWindow::onPlayTick() {

}

void MainWindow::seekToFrame(int frame_idx) {
  
}

void MainWindow::updateTimeLabel(int frame_idx,
                                 int total_frames,
                                 double fps) {
}

QPixmap MainWindow::matToQPixmap(const cv::Mat& mat_bgr) {
  if (mat_bgr.empty()) {
    return QPixmap();
  }
  cv::Mat mat_rgb;
  cv::cvtColor(mat_bgr, mat_rgb, cv::COLOR_BGR2RGB);
  QImage img(mat_rgb.data,
             mat_rgb.cols,
             mat_rgb.rows,
             static_cast<int>(mat_rgb.step),
             QImage::Format_RGB888);
  return QPixmap::fromImage(img.copy());
}

}  // namespace airsteady
