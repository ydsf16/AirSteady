#include "readme_dialog.hpp"

#include <QVBoxLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QFont>

namespace airsteady {

ReadmeDialog::ReadmeDialog(QWidget* parent)
    : QDialog(parent) {
  setWindowTitle(tr("使用说明"));
  resize(640, 420);

  auto* layout = new QVBoxLayout(this);

  auto* title = new QLabel(tr("AirSteady 使用说明"), this);
  QFont tf = title->font();
  tf.setPointSize(tf.pointSize() + 2);
  tf.setBold(true);
  title->setFont(tf);
  title->setAlignment(Qt::AlignCenter);
  layout->addWidget(title);

  layout->addSpacing(10);

  auto* intro = new QLabel(this);
  intro->setText(
      tr("欢迎使用 AirSteady！\n\n"
         "AirSteady 可以帮助你把抖动的航空视频稳定住，让画面始终盯住飞机。\n"
         "系统会在后台自动完成目标跟踪和运镜规划，通过逐帧裁切生成一段始终跟随飞机的新视频，"
         "从而达到显著的稳像效果。"));
  intro->setWordWrap(true);
  layout->addWidget(intro);

  layout->addSpacing(10);

  auto* usage = new QLabel(this);
  usage->setText(
      tr("<b>使用流程：</b><br>"
         "1. 点击左上角「打开视频」，选择你要处理的航空视频；<br>"
         "2. 打开后，程序会自动开始跟踪并规划运镜路径（当前 C++ Demo 仅做简单预览）；<br>"
         "3. 处理完成后，可以在左右两个窗口中预览原始视频与稳像结果；<br>"
         "4. 若未来版本集成参数调节与重新运镜，可在右侧调节参数后重新运镜；<br>"
         "5. 预览满意后，点击「导出视频」，即可导出稳像视频。"));
  usage->setWordWrap(true);
  layout->addWidget(usage);

  layout->addSpacing(10);

  auto* feedback = new QLabel(this);
  feedback->setText(
      tr("<b>问题反馈：</b><br>"
         "如果使用过程中遇到问题，可以点击主界面右上角的「问题反馈」按钮。\n"
         "当前 C++ 版本问题反馈入口仅为占位，后续可以接入你现有的反馈流程。"));
  feedback->setWordWrap(true);
  layout->addWidget(feedback);

  layout->addStretch(1);

  auto* btn_box = new QDialogButtonBox(QDialogButtonBox::Close, this);
  connect(btn_box, &QDialogButtonBox::rejected, this, &ReadmeDialog::reject);
  connect(btn_box, &QDialogButtonBox::accepted, this, &ReadmeDialog::accept);
  layout->addWidget(btn_box);
}

}  // namespace airsteady
