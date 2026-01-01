#pragma once

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>

#include "common/types.h"

namespace airsteady {

class ControlPanel : public QWidget {
  Q_OBJECT
 public:
  explicit ControlPanel(QWidget* parent = nullptr);

  StableParams GetParams() const;

  void SetParams(const StableParams& params);
  void SetCropKeepRatio(double alpha);
  void SetOffsetU(int value);
  void SetOffsetV(int value);

  QPushButton* recomputeButton();

 signals:
  void smoothChanged(double alpha);
  void cropEnableChanged(int value);
  void cropKeepRatioChanged(double keep_ratio);
  void offsetUChanged(int value);
  void offsetVChanged(int value);

 private slots:
  void onSmoothSliderChanged(int value);

  void onCropEnableChanged(int state);
  void onCropKeepRatioChanged(int value);
  void onOffsetUChanged(int value);
  void onOffsetVChanged(int value);

 private:
  QComboBox* track_class_combo_ = nullptr;

  // 稳定相关，需要重新做规划计算的步骤
  QSlider* smooth_slider_ = nullptr;
  QLabel* smooth_value_label_ = nullptr;
  QPushButton* recompute_btn_ = nullptr;

  // 裁切相关参数
  QCheckBox* crop_enable_check_box_ = nullptr;
  QSpinBox* crop_keep_ratio_edit_ = nullptr;
  QSpinBox* offset_u_edit_ = nullptr;
  QSpinBox* offset_v_edit_ = nullptr;
};

}  // namespace airsteady
