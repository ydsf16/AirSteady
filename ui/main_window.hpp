#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>

#include <opencv2/opencv.hpp>

#include "algorithm/processor.h"
#include "yolo/yolo_seg_detector.h"

class QLabel;
class QPushButton;
class QToolButton;
class QSlider;
class QStatusBar;
class QProgressBar;

namespace airsteady {

class VideoView;
class ControlPanel;

class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;

  bool eventFilter(QObject* obj, QEvent* event) override;

 protected:
  void closeEvent(QCloseEvent* event) override;

 private slots:
  void onOpenClicked();
  void onExportClicked();
  void onPlayPauseClicked();
  void onTimelineReleased();
  void onReadmeClicked();
  void onFeedbackClicked();
  void onContactClicked();

  // 参数调节相关的
  void onSmoothChanged(double alpha);
  void onCropKeeRatioChanged(double keep_ratio);

  void onVideoViewClicked(double x_norm, double y_norm);
  
  void onPlayTick();

  // 算法结果回调函数
  void OnTrackingResult(const FrameTrackingResult& res);
  void OnTrackFinished();
  void OnStablePlaneFinished();

 private:
  void buildUi();

  void OnReceiveTrackingResult(const FrameTrackingResultPreview& track_preview);
  void onReceivePreviewResult(const FramePreview& frame_preview);

 private:
  // 顶部
  QPushButton* open_btn_ = nullptr;
  QPushButton* export_btn_ = nullptr;
  QPushButton* readme_btn_ = nullptr;
  QPushButton* feedback_btn_ = nullptr;
  QPushButton* author_btn_ = nullptr;
  QLabel* file_label_ = nullptr;

  // 中部
  VideoView* raw_view_ = nullptr;
  VideoView* steady_view_ = nullptr;
  ControlPanel* control_panel_ = nullptr;

  // 底部播放控制
  QToolButton* play_btn_ = nullptr;
  QSlider* timeline_slider_ = nullptr;
  QLabel* time_label_ = nullptr;

  // 状态栏
  QStatusBar* status_bar_ = nullptr;
  QProgressBar* export_progress_ = nullptr;

  std::shared_ptr<Processor> video_processor_;

  bool preview_is_run_ = false;
};

}  // namespace airsteady
