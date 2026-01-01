#include "control_panel.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QString>

namespace airsteady {

ControlPanel::ControlPanel(QWidget* parent)
    : QWidget(parent) {
  auto* layout = new QVBoxLayout(this);

  auto* title = new QLabel(tr("参数调节"));
  title->setAlignment(Qt::AlignCenter);
  title->setStyleSheet("font-weight: bold;");
  layout->addWidget(title);

  layout->addSpacing(10);

  // ---------------- 跟踪对象：标签 + 下拉在同一行 ----------------
  auto* track_label = new QLabel(tr("跟踪对象"));
  track_class_combo_ = new QComboBox();
  track_class_combo_->addItem(tr("飞机"));

  auto* track_row = new QHBoxLayout();
  track_row->addWidget(track_label);
  track_row->addWidget(track_class_combo_, /*stretch=*/1);
  layout->addLayout(track_row);

  // ---------------- 稳定程度 ----------------
  layout->addSpacing(10);
  auto* smooth_label = new QLabel(tr("稳定程度"));

  smooth_slider_ = new QSlider(Qt::Horizontal);
  smooth_slider_->setRange(0, 100);
  smooth_slider_->setValue(100);

  auto* smooth_left_label = new QLabel(tr("跟随原片"));
  auto* smooth_right_label = new QLabel(tr("强力稳像"));
  smooth_value_label_ = new QLabel("1.0");

  // 稳定程度：一行里 左文字 + 滑条 + 右文字
  auto* smooth_row1 = new QHBoxLayout();
  smooth_row1->addWidget(smooth_left_label);
  smooth_row1->addWidget(smooth_slider_, 1);
  smooth_row1->addWidget(smooth_right_label);

  layout->addWidget(smooth_label);
  layout->addLayout(smooth_row1);
  layout->addWidget(smooth_value_label_);

  // 重算按钮
  recompute_btn_ = new QPushButton(tr("重新稳定"));
  layout->addWidget(recompute_btn_);

  // ---------------- 裁切相关 ----------------
  layout->addSpacing(20);

  // 开启裁切：标签 + 勾选框一行
  auto* enable_crop_label = new QLabel(tr("开启裁切"));
  // 关键：不要让标签横向撑开
  enable_crop_label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

  crop_enable_check_box_ = new QCheckBox();
  crop_enable_check_box_->setCheckState(Qt::Checked);
  auto* crop_enable_row = new QHBoxLayout();
  // 可以手动调节这行内部控件之间的间距
  crop_enable_row->setSpacing(6);  // 比默认略小一点
  crop_enable_row->addWidget(enable_crop_label);
  crop_enable_row->addWidget(crop_enable_check_box_);
  // 如果想整体靠左，可以再加个 stretch：
  crop_enable_row->addStretch();
  layout->addLayout(crop_enable_row);

  // 裁切保留比例：标签 + SpinBox 一行
  auto* crop_label = new QLabel(tr("裁切保留比例"));
  crop_keep_ratio_edit_ = new QSpinBox();
  crop_keep_ratio_edit_->setRange(0, 100);
  crop_keep_ratio_edit_->setSingleStep(1);
  crop_keep_ratio_edit_->setSuffix(" %");

  auto* crop_row = new QHBoxLayout();
  crop_row->addWidget(crop_label);
  crop_row->addWidget(crop_keep_ratio_edit_, 1);
  layout->addLayout(crop_row);

  layout->addSpacing(10);

  // ---------------- 偏移：标签 + SpinBox 一行 ----------------

  // 横向偏移
  auto* offset_u_label = new QLabel(tr("横向偏移"));
  offset_u_edit_ = new QSpinBox();
  offset_u_edit_->setRange(-4000, 4000);
  offset_u_edit_->setSingleStep(1);
  offset_u_edit_->setSuffix(" px");

  auto* offset_u_row = new QHBoxLayout();
  offset_u_row->addWidget(offset_u_label);
  offset_u_row->addWidget(offset_u_edit_, 1);
  layout->addLayout(offset_u_row);

  // 竖向偏移
  auto* offset_v_label = new QLabel(tr("竖向偏移"));
  offset_v_edit_ = new QSpinBox();
  offset_v_edit_->setRange(-4000, 4000);
  offset_v_edit_->setSingleStep(1);
  offset_v_edit_->setSuffix(" px");

  auto* offset_v_row = new QHBoxLayout();
  offset_v_row->addWidget(offset_v_label);
  offset_v_row->addWidget(offset_v_edit_, 1);
  layout->addLayout(offset_v_row);

  layout->addStretch(1);
  setFixedWidth(320);

  // 信号连接
  connect(smooth_slider_, &QSlider::valueChanged,
          this, &ControlPanel::onSmoothSliderChanged);
  connect(crop_enable_check_box_, &QCheckBox::stateChanged,
          this, &ControlPanel::onCropEnableChanged);
  connect(crop_keep_ratio_edit_, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ControlPanel::onCropKeepRatioChanged);
  connect(offset_u_edit_, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ControlPanel::onOffsetUChanged);
  connect(offset_v_edit_, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ControlPanel::onOffsetVChanged);
}

StableParams ControlPanel::GetParams() const {
  StableParams params;

  params.track_type = "air_plane";
  params.smooth_ratio = smooth_slider_->value() / 100.0;
  params.enable_crop =
      (crop_enable_check_box_->checkState() == Qt::CheckState::Checked);
  params.crop_keep_ratio = crop_keep_ratio_edit_->value() / 100.0;
  params.offset_u = offset_u_edit_->value();
  params.offset_v = offset_v_edit_->value();

  return params;
}

void ControlPanel::SetParams(const StableParams& params) {
  smooth_slider_->setValue(static_cast<int>(params.smooth_ratio * 100));
  smooth_value_label_->setText(
      QString::number(params.smooth_ratio, 'f', 2));

  // 勾选状态
  crop_enable_check_box_->setCheckState(
      params.enable_crop ? Qt::Checked : Qt::Unchecked);

  // 外部传入 0~1 小数
  crop_keep_ratio_edit_->setValue(
      static_cast<int>(params.crop_keep_ratio * 100.0 + 0.5));

  offset_u_edit_->setValue(params.offset_u);
  offset_v_edit_->setValue(params.offset_v);

  // 根据状态开关控件
  bool enabled = params.enable_crop;
  crop_keep_ratio_edit_->setEnabled(enabled);
  offset_u_edit_->setEnabled(enabled);
  offset_v_edit_->setEnabled(enabled);
}

void ControlPanel::SetCropKeepRatio(double alpha) {
  crop_keep_ratio_edit_->setValue(static_cast<int>(alpha * 100.0 + 0.5));
}

void ControlPanel::SetOffsetU(int value) {
  offset_u_edit_->setValue(value);
}

void ControlPanel::SetOffsetV(int value) {
  offset_v_edit_->setValue(value);
}

QPushButton* ControlPanel::recomputeButton() {
  return recompute_btn_;
}

void ControlPanel::onSmoothSliderChanged(int value) {
  double alpha = value / 100.0;
  smooth_value_label_->setText(QString::number(alpha, 'f', 2));
  emit smoothChanged(alpha);
}

void ControlPanel::onCropEnableChanged(int state) {
  bool enabled = (state == Qt::Checked);
  crop_keep_ratio_edit_->setEnabled(enabled);
  offset_u_edit_->setEnabled(enabled);
  offset_v_edit_->setEnabled(enabled);

  emit cropEnableChanged(state);
}

void ControlPanel::onCropKeepRatioChanged(int value) {
  if (crop_enable_check_box_->checkState() != Qt::Checked) {
    return;
  }
  double alpha = value / 100.0;
  emit cropKeepRatioChanged(alpha);
}

void ControlPanel::onOffsetUChanged(int value) {
  if (crop_enable_check_box_->checkState() != Qt::Checked) {
    return;
  }
  emit offsetUChanged(value);
}

void ControlPanel::onOffsetVChanged(int value) {
  if (crop_enable_check_box_->checkState() != Qt::Checked) {
    return;
  }
  emit offsetVChanged(value);
}

}  // namespace airsteady
