#pragma once

#include <QDialog>
#include <QString>
#include <QSpinBox>

#include "common/types.h"

class QComboBox;
class QLineEdit;

namespace airsteady {

class ExportDialog : public QDialog {
  Q_OBJECT
 public:
  ExportDialog(QWidget* parent, const std::string& video_name);

  ExportParams GetParams();

  void SetParams(const ExportParams& params);

 private slots:
  void onBrowse();
  void onAccept();

 private:
  QLineEdit* path_edit_ = nullptr;
  QComboBox* resolution_combo_ = nullptr;
  QSpinBox* bitrate_edit_ = nullptr;
};

}  // namespace airsteady
