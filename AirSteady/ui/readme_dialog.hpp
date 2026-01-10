#pragma once

#include <QDialog>

namespace airsteady {

class ReadmeDialog : public QDialog {
  Q_OBJECT
 public:
  explicit ReadmeDialog(QWidget* parent = nullptr);
};

}  // namespace airsteady
