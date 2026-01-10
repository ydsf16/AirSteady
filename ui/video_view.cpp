#include "ui/video_view.hpp"

#include <QPainter>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QFontMetrics>
#include <QMetaObject>

#include <algorithm>
#include <cmath>

#include <glog/logging.h>

namespace airsteady {

namespace {
inline double Clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}
}  // namespace

VideoView::VideoView(const QString& title, QWidget* parent)
    : QWidget(parent), title_(title) {
  setMinimumSize(320, 240);
  setMouseTracking(true);
}

void VideoView::SetTrackFrame(const FrameTrackingResultPreview& track_res) {
  // Deep copy image first (so we drop any dependency on caller's cv::Mat buffer).
  QImage img_copy = CvMatToQImageDeepCopy(track_res.proxy_bgr);

  DrawOverlay ov = BuildOverlayFromSegRes(track_res.seg_detect_res);

  {
    std::lock_guard<std::mutex> lk(mu_);
    mode_ = RenderMode::kTracking;
    img_ = std::move(img_copy);
    has_frame_ = !img_.isNull();
    overlay_ = std::move(ov);

    has_stable_ = false;
    global_offset_px_ = Eigen::Vector2d::Zero();
    stable_res_ = FrameStableResult{};
  }

  RequestRepaintAsync();
}

void VideoView::SetPreviewFrame(const FramePreview& frame_preview,
                                const Eigen::Vector2d& global_offset,
                                bool left) {
  QImage img_copy = CvMatToQImageDeepCopy(frame_preview.proxy_bgr);

  std::optional<DrawOverlay> ov;
  if (left) {
    // Left preview shows tracking overlay.
    ov = BuildOverlayFromSegRes(frame_preview.track_res.seg_detect_res);
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    mode_ = left ? RenderMode::kPreviewLeft : RenderMode::kPreviewRight;
    img_ = std::move(img_copy);
    has_frame_ = !img_.isNull();

    overlay_ = ov;

    global_offset_px_ = global_offset;
    stable_res_ = frame_preview.stable_res;
    has_stable_ = !left;  // only meaningful for right preview
  }

  RequestRepaintAsync();
}

void VideoView::ClearFrame() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    mode_ = RenderMode::kNone;
    img_ = QImage();
    has_frame_ = false;
    overlay_.reset();

    has_stable_ = false;
    global_offset_px_ = Eigen::Vector2d::Zero();
    stable_res_ = FrameStableResult{};
  }
  RequestRepaintAsync();
}

void VideoView::SetOverlay(const QString& text, bool center, const QColor& color) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    overlay_text_ = text;
    overlay_center_ = center;
    overlay_color_ = color.isValid() ? color : QColor(255, 255, 255);
    has_overlay_ = !overlay_text_.isEmpty();
  }
  RequestRepaintAsync();
}

void VideoView::ClearOverlay() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    overlay_text_.clear();
    has_overlay_ = false;
  }
  RequestRepaintAsync();
}

void VideoView::SetWarnOverlay(const QString& text, const QColor& color) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    warn_overlay_text_ = text;
    warn_overlay_color_ = color.isValid() ? color : QColor(255, 180, 0);
    has_warn_overlay_ = !warn_overlay_text_.isEmpty();
  }
  RequestRepaintAsync();
}

void VideoView::ClearWarnOverlay() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    warn_overlay_text_.clear();
    has_warn_overlay_ = false;
  }
  RequestRepaintAsync();
}

void VideoView::RequestRepaintAsync() {
  // Ensure repaint request happens on UI thread even if setters are called from workers.
  QMetaObject::invokeMethod(this, [this]() { this->update(); }, Qt::QueuedConnection);
}

QImage VideoView::CvMatToQImageDeepCopy(const cv::Mat& mat) {
  if (mat.empty()) return QImage();

  cv::Mat src = mat;
  if (!src.isContinuous()) {
    src = src.clone();
  }

  if (src.type() == CV_8UC3) {
    // OpenCV default: BGR
    QImage img(reinterpret_cast<const uchar*>(src.data),
               src.cols, src.rows,
               static_cast<int>(src.step),
               QImage::Format_BGR888);
    return img.copy();  // deep copy
  }

  if (src.type() == CV_8UC1) {
    QImage img(reinterpret_cast<const uchar*>(src.data),
               src.cols, src.rows,
               static_cast<int>(src.step),
               QImage::Format_Grayscale8);
    return img.copy();
  }

  // Fallback: convert to BGR8
  cv::Mat bgr;
  if (src.channels() == 4) {
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
  } else {
    src.convertTo(bgr, CV_8UC3);
  }

  QImage img(reinterpret_cast<const uchar*>(bgr.data),
             bgr.cols, bgr.rows,
             static_cast<int>(bgr.step),
             QImage::Format_BGR888);
  return img.copy();
}

VideoView::DrawOverlay VideoView::BuildOverlayFromSegRes(const SegDetectorRes& seg) {
  DrawOverlay ov;
  ov.frame_idx = seg.frame_idx;
  ov.time_ns = seg.time_ns;
  ov.timing = seg.timing;

  ov.dets.reserve(seg.yolo_objects.size());
  for (const auto& d : seg.yolo_objects) {
    DrawDet dd;
    dd.class_id = d.class_id;
    dd.score = d.score;
    dd.box = QRectF(d.box.x, d.box.y, d.box.width, d.box.height);
    dd.selected = false;
    ov.dets.push_back(dd);
  }

  ov.has_selected = seg.has_select_object;
  if (seg.has_select_object) {
    const auto& sb = seg.select_object.box;
    ov.selected_box = QRectF(sb.x, sb.y, sb.width, sb.height);

    // Mark selected if it matches exactly one of dets (best-effort).
    for (auto& dd : ov.dets) {
      if (dd.class_id == seg.select_object.class_id &&
          std::abs(dd.score - seg.select_object.score) < 1e-6f &&
          std::abs(dd.box.x() - ov.selected_box.x()) < 1e-3 &&
          std::abs(dd.box.y() - ov.selected_box.y()) < 1e-3 &&
          std::abs(dd.box.width() - ov.selected_box.width()) < 1e-3 &&
          std::abs(dd.box.height() - ov.selected_box.height()) < 1e-3) {
        dd.selected = true;
        break;
      }
    }
  }

  ov.good_pts.reserve(seg.good_pts_to_track.size());
  for (const auto& p : seg.good_pts_to_track) {
    ov.good_pts.emplace_back(p.x, p.y);
  }

  return ov;
}

QRectF VideoView::ComputeLetterboxRect(int img_w, int img_h) const {
  if (img_w <= 0 || img_h <= 0) return QRectF();

  const QRectF r = rect();
  const double vw = r.width();
  const double vh = r.height();

  const double sx = vw / static_cast<double>(img_w);
  const double sy = vh / static_cast<double>(img_h);
  const double s = std::min(sx, sy);

  const double dw = img_w * s;
  const double dh = img_h * s;

  const double x = (vw - dw) * 0.5;
  const double y = (vh - dh) * 0.5;

  return QRectF(x, y, dw, dh);
}

QPointF VideoView::MapImageToWidget(const QPointF& p_img, const QRectF& draw_rect,
                                   int img_w, int img_h) const {
  const double sx = draw_rect.width() / static_cast<double>(img_w);
  const double sy = draw_rect.height() / static_cast<double>(img_h);
  return QPointF(draw_rect.x() + p_img.x() * sx,
                 draw_rect.y() + p_img.y() * sy);
}

QRectF VideoView::MapImageRectToWidget(const QRectF& r_img, const QRectF& draw_rect,
                                       int img_w, int img_h) const {
  const QPointF tl = MapImageToWidget(QPointF(r_img.x(), r_img.y()), draw_rect, img_w, img_h);
  const QPointF br = MapImageToWidget(QPointF(r_img.x() + r_img.width(),
                                             r_img.y() + r_img.height()), draw_rect, img_w, img_h);
  return QRectF(tl, br).normalized();
}

void VideoView::PaintTitleAndOverlays(QPainter* painter, const QRectF& draw_rect) {
  painter->save();

  // Title bar
  {
    const int pad = 6;
    QFont f = painter->font();
    f.setPointSize(std::max(9, f.pointSize()));
    f.setBold(true);
    painter->setFont(f);

    const QString t = title_;
    QFontMetrics fm(f);
    const int th = fm.height() + 2 * pad;

    QRectF title_rect(0, 0, width(), th);
    painter->fillRect(title_rect, QColor(0, 0, 0, 120));
    painter->setPen(QColor(230, 230, 230));
    painter->drawText(title_rect.adjusted(pad, 0, -pad, 0),
                      Qt::AlignVCenter | Qt::AlignLeft, t);
  }

  // Overlay (center or top-left)
  QString overlay_text;
  bool overlay_center = true;
  QColor overlay_color;
  bool has_overlay = false;

  QString warn_text;
  QColor warn_color;
  bool has_warn = false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    overlay_text = overlay_text_;
    overlay_center = overlay_center_;
    overlay_color = overlay_color_;
    has_overlay = has_overlay_;

    warn_text = warn_overlay_text_;
    warn_color = warn_overlay_color_;
    has_warn = has_warn_overlay_;
  }

  if (has_overlay) {
    painter->setPen(overlay_color);
    painter->setBrush(Qt::NoBrush);

    QFont f = painter->font();
    f.setBold(true);
    painter->setFont(f);

    QRectF r = draw_rect;
    if (!overlay_center) {
      r = QRectF(draw_rect.x() + 10, draw_rect.y() + 10,
                 draw_rect.width() - 20, draw_rect.height() - 20);
      painter->fillRect(QRectF(r.x() - 6, r.y() - 6, 420, 32), overlay_bg_);
      painter->drawText(r, Qt::AlignLeft | Qt::AlignTop, overlay_text);
    } else {
      painter->fillRect(QRectF(draw_rect.center().x() - 260, draw_rect.center().y() - 20,
                               520, 40), overlay_bg_);
      painter->drawText(draw_rect, Qt::AlignCenter, overlay_text);
    }
  }

  if (has_warn) {
    painter->setPen(warn_color);
    QFont f = painter->font();
    f.setBold(true);
    painter->setFont(f);

    QRectF r(draw_rect.x() + 10, draw_rect.y() + draw_rect.height() - 50,
             draw_rect.width() - 20, 40);
    painter->fillRect(r.adjusted(-6, -4, 6, 4), QColor(0, 0, 0, 160));
    painter->drawText(r, Qt::AlignLeft | Qt::AlignVCenter, warn_text);
  }

  painter->restore();
}

void VideoView::PaintTrackingOverlay(QPainter* painter, const QRectF& draw_rect,
                                     int img_w, int img_h, const DrawOverlay& ov) {
  painter->save();

  // bboxes
  QPen pen_all(QColor(0, 220, 255), 2);
  QPen pen_sel(QColor(0, 255, 0), 3);

  painter->setBrush(Qt::NoBrush);

  for (const auto& d : ov.dets) {
    QRectF wr = MapImageRectToWidget(d.box, draw_rect, img_w, img_h);
    painter->setPen(d.selected ? pen_sel : pen_all);
    painter->drawRect(wr);

    // label
    QString label = QString("id=%1  s=%2")
                        .arg(d.class_id)
                        .arg(QString::number(d.score, 'f', 2));
    QRectF lr(wr.x(), wr.y() - 18, 160, 18);
    painter->fillRect(lr, QColor(0, 0, 0, 140));
    painter->setPen(QColor(240, 240, 240));
    painter->drawText(lr.adjusted(4, 0, -4, 0), Qt::AlignVCenter | Qt::AlignLeft, label);
  }

  // selected box highlight (in case it didn't match a det)
  if (ov.has_selected) {
    QRectF wr = MapImageRectToWidget(ov.selected_box, draw_rect, img_w, img_h);
    painter->setPen(pen_sel);
    painter->drawRect(wr);
  }

  // good points
  painter->setPen(Qt::NoPen);
  painter->setBrush(QColor(255, 220, 0));
  const double r = 2.5;
  for (const auto& p : ov.good_pts) {
    QPointF wp = MapImageToWidget(p, draw_rect, img_w, img_h);
    painter->drawEllipse(wp, r, r);
  }

  // timing text
  {
    painter->setPen(QColor(230, 230, 230));
    painter->setBrush(Qt::NoBrush);

    QString t = QString("frame=%1  yolo=%2ms  select=%3ms  gftt=%4ms  total=%5ms")
                    .arg(ov.frame_idx)
                    .arg(QString::number(ov.timing.yolo_ms, 'f', 2))
                    .arg(QString::number(ov.timing.select_object_ms, 'f', 2))
                    .arg(QString::number(ov.timing.gftt_ms, 'f', 2))
                    .arg(QString::number(ov.timing.total_ms, 'f', 2));

    QRectF tr(draw_rect.x() + 10, draw_rect.y() + 10, draw_rect.width() - 20, 24);
    painter->fillRect(tr.adjusted(-6, -3, 6, 3), QColor(0, 0, 0, 140));
    painter->drawText(tr, Qt::AlignLeft | Qt::AlignVCenter, t);
  }

  painter->restore();
}

void VideoView::PaintStabilizedView(QPainter* painter, const QRectF& draw_rect,
                                    const QImage& img, const Eigen::Vector2d& shift_px) {
  painter->save();

  // Paint black canvas inside draw_rect
  painter->fillRect(draw_rect, QColor(0, 0, 0));

  const int img_w = img.width();
  const int img_h = img.height();
  if (img_w <= 0 || img_h <= 0) {
    painter->restore();
    return;
  }

  const double sx = draw_rect.width() / static_cast<double>(img_w);
  const double sy = draw_rect.height() / static_cast<double>(img_h);

  const double tx = -shift_px.x() * sx;
  const double ty = -shift_px.y() * sy;

  // --------- 1) draw to screen (existing behavior) ----------
  painter->save();
  painter->setClipRect(draw_rect);
  painter->translate(tx, ty);
  painter->drawImage(draw_rect, img);
  painter->restore();

  // Draw shift text
  painter->setPen(QColor(230, 230, 230));
  QString s = QString("shift_px = ( %1 , %2 )")
                  .arg(QString::number(shift_px.x(), 'f', 2))
                  .arg(QString::number(shift_px.y(), 'f', 2));
  QRectF tr(draw_rect.x() + 10, draw_rect.y() + 10, draw_rect.width() - 20, 24);
  painter->fillRect(tr.adjusted(-6, -3, 6, 3), QColor(0, 0, 0, 140));
  painter->drawText(tr, Qt::AlignLeft | Qt::AlignVCenter, s);

  // --------- 2) debug record: offscreen render + VideoWriter (static) ----------
  // Toggle by environment variable to avoid touching UI plumbing:
  //   AIRSTEADY_STABLE_REC=1
  // Output:
  //   ./stable_debug.mp4   (relative to working dir)
  static bool s_enabled = true;
  static bool s_inited = false;
  static cv::VideoWriter s_writer;
  static int s_w = 0, s_h = 0;
  static int64_t s_frame_count = 0;

  if (!s_inited) {
    s_inited = true;
    const QByteArray env = qgetenv("AIRSTEADY_STABLE_REC");
    // s_enabled = (!env.isEmpty() && env != "0");

    if (s_enabled) {
      // Record full widget size for easiest viewing.
      const QSize widget_size = this->size();
      s_w = widget_size.width();
      s_h = widget_size.height();

      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      const double fps = 30.0;  // fixed debug fps
      const std::string out_path = "stable_debug.mp4";

      s_writer.open(out_path, fourcc, fps, cv::Size(s_w, s_h), true);
      if (!s_writer.isOpened()) {
        LOG(ERROR) << "[VideoView][StableRec] Failed to open " << out_path;
        s_enabled = false;
      } else {
        LOG(INFO) << "[VideoView][StableRec] Recording to " << out_path
                  << " size=" << s_w << "x" << s_h << " fps=" << fps;
      }
    }
  }

  if (s_enabled && s_writer.isOpened()) {
    // If user resizes the widget mid-run, stop to avoid broken stream.
    const QSize widget_size = this->size();
    if (widget_size.width() != s_w || widget_size.height() != s_h) {
      LOG(WARNING) << "[VideoView][StableRec] Widget resized, stop recording. "
                   << "old=" << s_w << "x" << s_h
                   << " new=" << widget_size.width() << "x" << widget_size.height();
      s_writer.release();
      s_enabled = false;
    } else {
      // Offscreen canvas = what user sees (full widget).
      QImage canvas(widget_size, QImage::Format_ARGB32_Premultiplied);
      canvas.fill(QColor(20, 20, 20));  // same background as paintEvent()

      QPainter p2(&canvas);
      p2.setRenderHint(QPainter::Antialiasing, true);

      // Re-render the stabilized view into canvas (same logic, but using p2)
      // NOTE: draw_rect is in widget coords already.
      p2.fillRect(draw_rect, QColor(0, 0, 0));

      p2.save();
      p2.setClipRect(draw_rect);
      p2.translate(tx, ty);
      p2.drawImage(draw_rect, img);
      p2.restore();

      p2.setPen(QColor(230, 230, 230));
      p2.fillRect(tr.adjusted(-6, -3, 6, 3), QColor(0, 0, 0, 140));
      p2.drawText(tr, Qt::AlignLeft | Qt::AlignVCenter, s);
      p2.end();

      // QImage -> cv::Mat(BGR)
      QImage argb = canvas.convertToFormat(QImage::Format_ARGB32);
      cv::Mat bgra(argb.height(), argb.width(), CV_8UC4,
                   const_cast<uchar*>(argb.bits()),
                   static_cast<size_t>(argb.bytesPerLine()));
      cv::Mat bgr;
      cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

      s_writer.write(bgr);
      ++s_frame_count;

      // Optional: auto-stop after N frames to avoid huge files.
      // if (s_frame_count >= 300) { s_writer.release(); s_enabled = false; }
      if ((s_frame_count % 60) == 0) {
        LOG(INFO) << "[VideoView][StableRec] wrote frames=" << s_frame_count;
      }
    }
  }

  painter->restore();
}

void VideoView::paintEvent(QPaintEvent* /*event*/) {
  LOG_EVERY_N(INFO, 60) << "VideoView paintEvent: " << title_.toStdString()
                        << " size=" << width() << "x" << height();

  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.fillRect(rect(), QColor(20, 20, 20));

  // Snapshot state under lock (avoid holding lock while drawing).
  RenderMode mode = RenderMode::kNone;
  QImage img;
  bool has_frame = false;
  std::optional<DrawOverlay> ov;
  Eigen::Vector2d global_offset = Eigen::Vector2d::Zero();
  FrameStableResult stable_res;
  bool has_stable = false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    mode = mode_;
    img = img_;
    has_frame = has_frame_;
    ov = overlay_;
    global_offset = global_offset_px_;
    stable_res = stable_res_;
    has_stable = has_stable_;
  }

  if (!has_frame || img.isNull()) {
    // No frame: show title and overlays only.
    last_draw_rect_ = QRectF();
    PaintTitleAndOverlays(&painter, QRectF(0, 0, width(), height()));
    return;
  }

  const int img_w = img.width();
  const int img_h = img.height();

  QRectF draw_rect = ComputeLetterboxRect(img_w, img_h);
  last_draw_rect_ = draw_rect;

  // Base image
  if (mode == RenderMode::kPreviewRight && has_stable) {
    // Right preview: stabilized render by translation onto canvas.
    const Eigen::Vector2d shift = global_offset + Eigen::Vector2d(stable_res.delta_x, stable_res.delta_y);
    PaintStabilizedView(&painter, draw_rect, img, shift);
  } else {
    // Tracking / Left preview: draw raw frame fit to view.
    painter.drawImage(draw_rect, img);

    if (ov.has_value()) {
      PaintTrackingOverlay(&painter, draw_rect, img_w, img_h, ov.value());
    }
  }

  // Title + overlays always on top
  PaintTitleAndOverlays(&painter, draw_rect);
}

void VideoView::mousePressEvent(QMouseEvent* event) {
  // Optional: Click-to-select behavior can be expanded later.
  // Currently: emit normalized point rect (tiny) so upstream can decide.
  if (event == nullptr) return;

  QRectF draw_rect;
  QImage img;
  {
    std::lock_guard<std::mutex> lk(mu_);
    draw_rect = last_draw_rect_;
    img = img_;
  }
  if (img.isNull() || draw_rect.isEmpty()) return;

  const QPointF p = event->pos();
  if (!draw_rect.contains(p)) return;

  // Convert widget point -> image normalized.
  const double nx = (p.x() - draw_rect.x()) / draw_rect.width();
  const double ny = (p.y() - draw_rect.y()) / draw_rect.height();

  const double clx = Clamp(nx, 0.0, 1.0);
  const double cly = Clamp(ny, 0.0, 1.0);

  QRectF tiny(clx, cly, 0.001, 0.001);
  emit TargetRectSelected(tiny);
}

}  // namespace airsteady
