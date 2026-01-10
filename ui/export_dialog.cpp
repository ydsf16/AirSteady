#include "export_dialog.hpp"

#include <algorithm>

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QTemporaryFile>

namespace airsteady {
namespace {

constexpr int kMinBitrateMbps = 1;
constexpr int kMaxBitrateMbps = 200;
constexpr int kDefaultBitrateMbps = 20;

// export_resolution encoding (int):
//   0 = Auto
//   1 = 4K
//   2 = 2K
//   3 = 1080p
//   4 = 720p
enum ResolutionCode : int {
  kResAuto = 0,
  kRes4k = 1,
  kRes2k = 2,
  kRes1080p = 3,
  kRes720p = 4,
};

QString EnsureMp4Suffix(QString path) {
  path = path.trimmed();
  if (path.isEmpty()) {
    return path;
  }
  if (!path.endsWith(".mp4", Qt::CaseInsensitive)) {
    path += ".mp4";
  }
  return path;
}

QString DefaultExportNameFromVideoName(const QString& video_name) {
  // Keep it simple and predictable: "<basename>_export.mp4"
  const QFileInfo fi(video_name);
  const QString base = fi.completeBaseName().isEmpty() ? QString("output") : fi.completeBaseName();
  return EnsureMp4Suffix(base + "_export.mp4");
}

int FindComboIndexByCode(const QComboBox* combo, int code) {
  if (combo == nullptr) {
    return -1;
  }
  for (int i = 0; i < combo->count(); ++i) {
    if (combo->itemData(i).toInt() == code) {
      return i;
    }
  }
  return -1;
}

}  // namespace

ExportDialog::ExportDialog(QWidget* parent, const std::string& video_name)
    : QDialog(parent) {
  setWindowTitle(tr("Export"));

  // -------- Path --------
  path_edit_ = new QLineEdit(this);
  path_edit_->setPlaceholderText(tr("Choose output .mp4 path"));

  auto* browse_btn = new QPushButton(tr("Browse..."), this);
  connect(browse_btn, &QPushButton::clicked, this, &ExportDialog::onBrowse);

  auto* path_row = new QHBoxLayout();
  path_row->addWidget(path_edit_, /*stretch=*/1);
  path_row->addWidget(browse_btn);

  // Default output file name (no directory yet; user can Browse).
  const QString video_q = QString::fromStdString(video_name);
  path_edit_->setText(DefaultExportNameFromVideoName(video_q));

  // -------- Bitrate (Mbps) --------
  bitrate_edit_ = new QSpinBox(this);
  bitrate_edit_->setRange(kMinBitrateMbps, kMaxBitrateMbps);
  bitrate_edit_->setValue(kDefaultBitrateMbps);
  bitrate_edit_->setSuffix(tr(" M"));
  bitrate_edit_->setToolTip(tr("Video bitrate in Mbps (e.g., 20M)."));

  // -------- Resolution --------
  resolution_combo_ = new QComboBox(this);
  resolution_combo_->addItem(tr("Auto"), static_cast<int>(kResAuto));
  // resolution_combo_->addItem(tr("4K (3840x2160)"), static_cast<int>(kRes4k));
  // resolution_combo_->addItem(tr("2K (2560x1440)"), static_cast<int>(kRes2k));
  // resolution_combo_->addItem(tr("1080p (1920x1080)"), static_cast<int>(kRes1080p));
  // resolution_combo_->addItem(tr("720p (1280x720)"), static_cast<int>(kRes720p));
  resolution_combo_->setCurrentIndex(0);

  // -------- Buttons --------
  auto* buttons =
      new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
  connect(buttons, &QDialogButtonBox::accepted, this, &ExportDialog::onAccept);
  connect(buttons, &QDialogButtonBox::rejected, this, &ExportDialog::reject);

  // -------- Layout --------
  auto* form = new QFormLayout();
  form->addRow(tr("Export path:"), path_row);
  form->addRow(tr("Bitrate:"), bitrate_edit_);
  form->addRow(tr("Resolution:"), resolution_combo_);

  auto* root = new QVBoxLayout(this);
  root->addLayout(form);
  root->addWidget(buttons);

  setLayout(root);
}

ExportParams ExportDialog::GetParams() {
  ExportParams p;

  const QString path = EnsureMp4Suffix(path_edit_ ? path_edit_->text() : QString());
  p.export_path = path.toStdString();

  const int bitrate_mbps = bitrate_edit_ ? bitrate_edit_->value() : 0;
  p.export_bitrate = static_cast<double>(bitrate_mbps) * 1000 * 1000;

  const int code = resolution_combo_ ? resolution_combo_->currentData().toInt() : 0;
  p.export_resolution = code;

  return p;
}

void ExportDialog::SetParams(const ExportParams& params) {
  if (path_edit_ != nullptr) {
    path_edit_->setText(EnsureMp4Suffix(QString::fromStdString(params.export_path)));
  }

  if (bitrate_edit_ != nullptr) {
    // Your param is double, UI is integer Mbps. We round to nearest int.
    const double bitrate_mb = params.export_bitrate / (1024 * 1024);
    const int bitrate_mbps = static_cast<int>(std::lround(bitrate_mb));
    const int clamped = std::clamp(bitrate_mbps, kMinBitrateMbps, kMaxBitrateMbps);
    bitrate_edit_->setValue(clamped);
  }

  if (resolution_combo_ != nullptr) {
    const int idx = FindComboIndexByCode(resolution_combo_, params.export_resolution);
    if (idx >= 0) {
      resolution_combo_->setCurrentIndex(idx);
    } else {
      resolution_combo_->setCurrentIndex(0);  // Auto fallback (the safest lie).
    }
  }
}

void ExportDialog::onBrowse() {
  // Use save file dialog so users can pick folder + filename in one go.
  const QString current = path_edit_ ? path_edit_->text().trimmed() : QString();
  const QString suggested = current.isEmpty() ? QString("output.mp4") : current;

  const QString chosen =
      QFileDialog::getSaveFileName(this,
                                   tr("Choose export file"),
                                   suggested,
                                   tr("MP4 Video (*.mp4)"));
  if (chosen.isEmpty()) {
    return;
  }

  if (path_edit_ != nullptr) {
    path_edit_->setText(EnsureMp4Suffix(chosen));
  }
}

void ExportDialog::onAccept() {
  ExportParams p = GetParams();

  // Normalize & ensure suffix.
  QString out_path = QString::fromStdString(p.export_path).trimmed();
  if (out_path.isEmpty()) {
    QMessageBox::warning(this, tr("Invalid"), tr("Export path is empty."));
    return;
  }
  if (!out_path.endsWith(".mp4", Qt::CaseInsensitive)) {
    out_path += ".mp4";
  }
  p.export_path = out_path.toStdString();

  // Bitrate check.
  if (p.export_bitrate <= 0.0) {
    QMessageBox::warning(this, tr("Invalid"), tr("Bitrate must be greater than 0."));
    return;
  }

  const QFileInfo out_info(out_path);

  // Path must not be a directory.
  if (out_info.exists() && out_info.isDir()) {
    QMessageBox::warning(this, tr("Invalid"), tr("Export path points to a directory."));
    return;
  }

  // Ensure parent directory exists (or create it).
  const QString dir_path = out_info.absolutePath();
  if (dir_path.isEmpty()) {
    QMessageBox::warning(this, tr("Invalid"), tr("Export directory is invalid."));
    return;
  }

  QDir dir(dir_path);
  if (!dir.exists()) {
    if (!dir.mkpath(".")) {
      QMessageBox::warning(
          this, tr("Invalid"),
          tr("Export directory does not exist and cannot be created:\n%1").arg(dir_path));
      return;
    }
  }

  // Check directory is writable by creating a temp file in it.
  {
    QTemporaryFile tmp(dir.filePath("airsteady_write_test_XXXXXX.tmp"));
    tmp.setAutoRemove(true);
    if (!tmp.open()) {
      QMessageBox::warning(
          this, tr("Invalid"),
          tr("Export directory is not writable:\n%1").arg(dir_path));
      return;
    }
  }

  // If file exists, confirm overwrite.
  if (out_info.exists() && out_info.isFile()) {
    const auto reply = QMessageBox::question(
        this, tr("Overwrite?"),
        tr("File already exists:\n%1\n\nOverwrite it?").arg(out_path),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (reply != QMessageBox::Yes) {
      return;
    }
  }

  // Optional: reject empty filename like ".../.mp4"
  if (out_info.fileName().trimmed().isEmpty() || out_info.fileName() == ".mp4") {
    QMessageBox::warning(this, tr("Invalid"), tr("Export filename is invalid."));
    return;
  }

  accept();
}

}  // namespace airsteady
