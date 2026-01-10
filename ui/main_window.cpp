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
#include "common/file_utils.hpp"

#include <glog/logging.h>

namespace airsteady {

const int kMaxProxyResolution = 1080; // 1080p.

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("AirSteady 航迹稳拍 - Beta"));
  resize(1400, 800);

  buildUi();
}

MainWindow::~MainWindow() {
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
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
  play_btn_ = new QToolButton(this);
  play_btn_->setText(QStringLiteral("▶"));

  timeline_slider_ = new QSlider(Qt::Horizontal, this);
  timeline_slider_->setRange(0, 0);
  timeline_slider_->setValue(0);
  time_label_ = new QLabel(QStringLiteral("00:00 / 00:00"), this);

  auto* bottom_bar = new QWidget(this);
  auto* bottom_layout = new QHBoxLayout(bottom_bar);
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
  connect(timeline_slider_, &QSlider::sliderReleased,
          this, &MainWindow::onTimelineReleased);
  
  // 参数变化相关
  connect(control_panel_, &ControlPanel::smoothChanged,
          this, &MainWindow::onSmoothChanged);
  connect(control_panel_, &ControlPanel::cropKeepRatioChanged,
          this, &MainWindow::onCropKeeRatioChanged);

  connect(timeline_slider_, &QSlider::valueChanged, this, &MainWindow::onTimelineSliderValueChanged);
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

  if (!IsFileExist(path.toStdString())) {
    // TODO: Qmessage, bad image.
    return;
  }

  std::string exe_folder;
  if (!GetExeFolder(&exe_folder)) {
    LOG(FATAL) << "Failed to get exe folder!!!";
  }
  Processor::Config config;
  config.video_path = path.toStdString();
  config.system_params.work_folder = exe_folder + "/work_folder";
  CreateFolder(config.system_params.work_folder);

  // TODO: Oth get from ui.
  auto new_video_processor = std::make_shared<Processor>(config);
  std::string err;
  if (!new_video_processor->TryOpenVideo(&err)) {
    QMessageBox::warning(
      this,
      tr("打开失败"),
      tr("视频文件异常，请检查视频文件是否正常!"),
      tr(err.c_str()));
    return;
  }
  
  // Stop old video processes.
  if (video_processor_) {
    video_processor_->StopAll();
  }

  // Assign to video_processor_.
  video_processor_ = new_video_processor;

  // TODO: Callbacks.
  video_processor_->AddTrackingResultCallback(
    [this](const FrameTrackingResultPreview& res) {
      OnReceiveTrackingResult(res);
    });

  video_processor_->AddPreviewCallback(
    [this](const FramePreview& res) {
      onReceivePreviewResult(res);
    });
  
  video_processor_->AddPreviewDoneCallback(
    [this]() {
      onPreviewDone();
    });
  
  // Export related.
  video_processor_->AddExportCallback([this](int frame_idx) {
      onExportCallback(frame_idx);
    });
  video_processor_->AddExportDoneCallback([this]() {
      onExportDone();
    });

  // Start tracking!!!
  video_processor_->StartTracking(&err);
}

void MainWindow::onExportClicked() {
  if (video_processor_ == nullptr) {
    QMessageBox::information(
      this,
      tr("失败"),
      tr("请先打开视频"));
    return;
  }
  if (video_processor_->status() != Processor::Status::kStabilized) {
    return;
  }

  const std::string video_path = video_processor_->config().video_path;
  std::string video_name;
  GetVideoName(video_path, &video_name);
  const VideoInfo video_info = video_processor_->GetVideoInfo();

  // Initial export param.
  ExportParams init_export_param = video_processor_->config().export_params;
  init_export_param.export_path = GetDesktopPath() + "/" + video_name + "_airsteady.mp4";
  init_export_param.export_bitrate = video_info.bitrate;
  LOG(INFO) << "Export bitreate: " << video_info.bitrate;

  ExportDialog dlg(this, video_name);
  dlg.SetParams(init_export_param);

  if (dlg.exec() != QDialog::Accepted) {
    return;
  }

  ExportParams set_params = dlg.GetParams();
  LOG(INFO) << "set_export_path: "  << std::fixed 
            << set_params.export_path 
            << ", bitrate: "  << set_params.export_bitrate
            << ", export_resolution: " << set_params.export_resolution; 

  video_processor_->SetExportParams(set_params);
  video_processor_->StartExport();
}

void MainWindow::onPlayPauseClicked() {
  if (!video_processor_) {return; }

  LOG(INFO) << "onPlayPauseClicked: " << ", " << (int)(video_processor_->status());
  if (video_processor_->status() != Processor::Status::kStabilized) {
    return;
  }

  if (preview_is_run_ == false) {
    LOG(INFO) <<  "Start preview";
    video_processor_->StartPreview();
    preview_is_run_ = true;
    play_btn_->setText(QStringLiteral("||"));
  } else {
    video_processor_->HoldPreview();
    LOG(INFO) <<  "Stop preview";
    preview_is_run_ = false;
    play_btn_->setText(QStringLiteral("▶"));
  }
}

void MainWindow::onTimelineReleased() {
  LOG(INFO) << "onTimelineReleased 1: ";
  if (video_processor_->status() != Processor::Status::kStabilized) {
    return;
  }
  LOG(INFO) << "onTimelineReleased 2: ";

  if (preview_is_run_ == true) {
    return;
  }

  LOG(INFO) << "onTimelineReleased 3: ";

  int frame_idx = timeline_slider_->value();
  LOG(INFO) << "Seek frame: " << frame_idx;

  video_processor_->SeekAndPreviewOnce(frame_idx);
  LOG(INFO) << "Seek frame2: " << frame_idx;
}

void MainWindow::onReadmeClicked() {
  ReadmeDialog dlg(this);
  dlg.exec();
}

void MainWindow::onFeedbackClicked() {
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
}

void MainWindow::onCropKeeRatioChanged(double keep_ratio) {
  qDebug() << "[UI] 裁切保留比例更新:" << keep_ratio;
}

void MainWindow::onVideoViewClicked(double /*x_norm*/,
                                    double /*y_norm*/) {
}

void MainWindow::onPlayTick() {

}

void MainWindow::OnTrackingResult(const FrameTrackingResult& res) {

}

void MainWindow::OnTrackFinished() {
  LOG(INFO) << "Main window receive ontrack finished.";
}

void MainWindow::OnStablePlaneFinished() {
}

void MainWindow::onPreviewDone() {
  LOG(WARNING) << "onPreviewDone";

  // video_processor_->SeekAndPreviewOnce(0);
  // video_processor_->HoldPreview();

  LOG(INFO) << "Stop preview";
  preview_is_run_ = false;
  play_btn_->setText(QStringLiteral("▶"));
}

void MainWindow::onTimelineSliderValueChanged(int value) {
  if (!video_processor_) {return; }

  if (video_processor_->status() != Processor::Status::kStabilized) {
    return;
  }
  if (preview_is_run_ == true) {
    return;
  }

  int frame_idx = timeline_slider_->value();
  if (frame_idx % 5 == 0) {
    video_processor_->SeekAndPreviewOnce(frame_idx);
  }
} 

void MainWindow::onExportCallback(int frame_idx) {
  auto video_info = video_processor_->GetVideoInfo();
  double progress_ratio = static_cast<double>(frame_idx) / video_info.num_frames;
  steady_view_->UpdateProgressBar(progress_ratio);
}

void MainWindow::onExportDone() {
  steady_view_->ClearProgressBar();
}

void MainWindow::OnReceiveTrackingResult(const FrameTrackingResultPreview& track_preview) {
  // Copy only what we need (Mat lifetime: VideoView will deep copy; here we pass by value in lambda).
  const FrameTrackingResultPreview preview_copy = track_preview;

  // raw_view_->SetTrackFrame(preview_copy);

  QMetaObject::invokeMethod(this, [this, preview_copy]() {
    if (!video_processor_) {
      LOG(WARNING) << "No video_processor_!";
      return;
    }
    if (!raw_view_) {
      LOG(WARNING) << "No raw_view_!";
      return;
    }
    if (video_processor_->status() != Processor::Status::kTracking) {
      return;
    }

    raw_view_->SetTrackFrame(preview_copy);

    const VideoInfo video_info = video_processor_->GetVideoInfo();
    const int total = std::max(0, static_cast<int>(video_info.num_frames));
    const int idx = preview_copy.seg_detect_res.frame_idx;

    if (timeline_slider_) {
      const int max_idx = std::max(0, total - 1);
      timeline_slider_->setRange(0, max_idx);
      timeline_slider_->setValue(std::clamp(idx, 0, max_idx));
    }

    if (time_label_) {
      double ratio = 0.0;
      if (total > 0) ratio = static_cast<double>(idx) / static_cast<double>(total);

      std::ostringstream ss;
      ss << "frame " << idx << "/" << std::max(0, total - 1)
         << " (" << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%)";
      time_label_->setText(QString::fromStdString(ss.str()));
    }
  }, Qt::QueuedConnection);
}

static int SafeClampFrameIdx(int idx, int total) {
  if (total <= 0) return 0;
  const int max_idx = std::max(0, total - 1);
  return std::clamp(idx, 0, max_idx);
}

static QString FormatFrameProgress(int idx, int total) {
  const int max_idx = std::max(0, total - 1);
  double ratio = 0.0;
  if (total > 0) ratio = static_cast<double>(idx) / static_cast<double>(total);

  std::ostringstream ss;
  ss << "frame " << idx << "/" << max_idx << " ("
     << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%)";
  return QString::fromStdString(ss.str());
}

void MainWindow::onReceivePreviewResult(const FramePreview& frame_preview) {
  QPointer<MainWindow> self(this);
  FramePreview preview_copy = frame_preview;  // 若包含 Mat 且复用缓冲，同理可 clone

  QMetaObject::invokeMethod(
      this,
      [this, preview = std::move(preview_copy)]() mutable {
        if (!video_processor_) return;

        if (raw_view_) {
          raw_view_->SetPreviewFrame(preview, Eigen::Vector2d::Zero(), true);
        }
        if (steady_view_) {
          steady_view_->SetPreviewFrame(preview, Eigen::Vector2d::Zero(), false);
        }

        const VideoInfo video_info = video_processor_->GetVideoInfo();
        const int total = std::max(0, static_cast<int>(video_info.num_frames));
        const int idx = preview.frame_idx;

        if (timeline_slider_) {
          const int max_idx = std::max(0, total - 1);
          timeline_slider_->setRange(0, max_idx);
          timeline_slider_->setValue(std::clamp(idx, 0, max_idx));
        }

        if (time_label_) {
          double ratio = 0.0;
          if (total > 0) ratio = static_cast<double>(idx) / static_cast<double>(total);

          std::ostringstream ss;
          ss << "frame " << idx << "/" << std::max(0, total - 1)
            << " (" << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%)";
          time_label_->setText(QString::fromStdString(ss.str()));
        }
      },
      Qt::QueuedConnection);
}

}  // namespace airsteady
