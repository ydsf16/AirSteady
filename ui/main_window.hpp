#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>

#include <opencv2/opencv.hpp>

class QLabel;
class QPushButton;
class QToolButton;
class QSlider;
class QStatusBar;
class QProgressBar;

namespace airsteady {

class VideoView;
class ControlPanel;

enum class AppState {
  kIdle = 0,
  kVideoLoaded,
  kPlaying,
  kPaused,
  kExporting,
};

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
  void onPrevClicked();
  void onTimelineReleased();
  void onReadmeClicked();
  void onFeedbackClicked();
  void onContactClicked();

  // 参数调节相关的
  void onSmoothChanged(double alpha);
  void onCropKeeRatioChanged(double keep_ratio);

  void onVideoViewClicked(double x_norm, double y_norm);

  void onPlayTick();

 private:
  void buildUi();
  void resetPreview();
  void resetVideoViews();
  void updateState(AppState new_state);

  void togglePlayPause();
  void seekToFrame(int frame_idx);
  void updateTimeLabel(int frame_idx,
                       int total_frames,
                       double fps);

  // BGR Mat -> QPixmap
  QPixmap matToQPixmap(const cv::Mat& mat_bgr);

 private:
  AppState state_ = AppState::kIdle;

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
  QToolButton* prev_btn_ = nullptr;
  QToolButton* play_btn_ = nullptr;
  QSlider* timeline_slider_ = nullptr;
  QLabel* time_label_ = nullptr;

  // 状态栏
  QStatusBar* status_bar_ = nullptr;
  QProgressBar* export_progress_ = nullptr;
};

}  // namespace airsteady
