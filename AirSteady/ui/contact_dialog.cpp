#include "contact_dialog.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QFont>
#include <QPixmap>
#include <QDir>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QImageReader>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace airsteady {
namespace {

static QPixmap LoadPixmapViaQtOrOpenCV(const QString& relative_path) {
  const QString base_dir = QCoreApplication::applicationDirPath();
  const QString full_path = QDir(base_dir).filePath(relative_path);

  // 1) Try Qt first (preferred)
  {
    QImageReader reader(full_path);
    reader.setAutoTransform(true);
    const QImage img = reader.read();
    if (!img.isNull()) {
      return QPixmap::fromImage(img);
    }
  }

  // 2) Fallback to OpenCV
  cv::Mat bgr = cv::imread(full_path.toStdString(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    return QPixmap();
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  // Make a deep copy owned by shared_ptr to ensure lifetime beyond this function.
  auto owned = std::make_shared<cv::Mat>(rgb);

  QImage qimg(owned->data,
              owned->cols,
              owned->rows,
              static_cast<int>(owned->step),
              QImage::Format_RGB888);

  // Ensure QImage keeps the backing store alive.
  QImage qimg_copy = qimg.copy();  // safest: detach from cv::Mat memory
  return QPixmap::fromImage(qimg_copy);
}

QPixmap loadQrPixmap(const QString& relative_path) {
  QPixmap pix = LoadPixmapViaQtOrOpenCV(relative_path);
  if (pix.isNull()) {
    return QPixmap();
  }
  const int size = 180;
  return pix.scaled(size, size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

}  // namespace

ContactDialog::ContactDialog(QWidget* parent)
    : QDialog(parent) {
  setWindowTitle(tr("联系我们"));
  resize(680, 480);

  auto* layout = new QVBoxLayout(this);

  auto* title = new QLabel(tr("关于作者"), this);
  QFont tf = title->font();
  tf.setPointSize(tf.pointSize() + 2);
  tf.setBold(true);
  title->setFont(tf);
  title->setAlignment(Qt::AlignCenter);
  layout->addWidget(title);

  layout->addSpacing(10);

  auto* intro = new QLabel(this);
  intro->setText(
      tr("<b>关于作者</b><br>"
         "我是 AirSteady 的作者[小葡萄爸爸]，一名长期从事机器人、计算机视觉等方向的研发工程师。<br>"
         "平时喜欢把复杂技术做成真正好用的工具，AirSteady 就是把技术和个人兴趣结合的一次尝试。<br><br>"
         "<b>更多了解与联系</b><br>"
         "欢迎通过下面的渠道交流技术、产品想法，或者聊聊飞机和旅行：<br>"
         "• 知乎："
         "<a href=\"https://www.zhihu.com/people/DongShengYang/posts/posts_by_votes\">"
         "https://www.zhihu.com/people/DongShengYang/posts/posts_by_votes</a><br>"
         "• GitHub："
         "<a href=\"https://github.com/ydsf16\">https://github.com/ydsf16</a><br>"));
  intro->setWordWrap(true);
  intro->setOpenExternalLinks(true);
  intro->setTextInteractionFlags(Qt::TextBrowserInteraction);
  layout->addWidget(intro);

  layout->addSpacing(10);

  auto* qr_row = new QHBoxLayout();
  qr_row->addStretch(1);

  // 抖音
  auto* douyin_col = new QVBoxLayout();
  auto* douyin_title = new QLabel(tr("抖音"), this);
  douyin_title->setAlignment(Qt::AlignCenter);
  auto* douyin_label = new QLabel(this);
  douyin_label->setAlignment(Qt::AlignCenter);

  QPixmap douyin_qr = loadQrPixmap("assets/douyin_qr.jpg");
  if (!douyin_qr.isNull()) {
    douyin_label->setPixmap(douyin_qr);
  } else {
    douyin_label->setText(tr("请在 assets/douyin_qr.jpg 放置抖音二维码"));
    douyin_label->setWordWrap(true);
  }

  auto* douyin_hint =
      new QLabel(tr("扫描二维码\n关注抖音账号"), this);
  douyin_hint->setAlignment(Qt::AlignCenter);
  douyin_hint->setWordWrap(true);

  douyin_col->addWidget(douyin_title);
  douyin_col->addWidget(douyin_label);
  douyin_col->addWidget(douyin_hint);

  // 微信
  auto* wechat_col = new QVBoxLayout();
  auto* wechat_title = new QLabel(tr("微信"), this);
  wechat_title->setAlignment(Qt::AlignCenter);
  auto* wechat_label = new QLabel(this);
  wechat_label->setAlignment(Qt::AlignCenter);

  QPixmap wechat_qr = loadQrPixmap("assets/wechat_qr.jpg");
  if (!wechat_qr.isNull()) {
    wechat_label->setPixmap(wechat_qr);
  } else {
    wechat_label->setText(tr("请在 assets/wechat_qr.jpg 放置微信二维码"));
    wechat_label->setWordWrap(true);
  }

  auto* wechat_hint =
      new QLabel(tr("扫描二维码\n添加微信好友（备注 “AirSteady”）"), this);
  wechat_hint->setAlignment(Qt::AlignCenter);
  wechat_hint->setWordWrap(true);

  wechat_col->addWidget(wechat_title);
  wechat_col->addWidget(wechat_label);
  wechat_col->addWidget(wechat_hint);

  qr_row->addLayout(douyin_col);
  qr_row->addSpacing(40);
  qr_row->addLayout(wechat_col);
  qr_row->addStretch(1);

  layout->addLayout(qr_row);
  layout->addStretch(1);

  auto* btn_box = new QDialogButtonBox(QDialogButtonBox::Close, this);
  connect(btn_box, &QDialogButtonBox::rejected, this, &ContactDialog::reject);
  connect(btn_box, &QDialogButtonBox::accepted, this, &ContactDialog::accept);
  layout->addWidget(btn_box);
}

}  // namespace airsteady
