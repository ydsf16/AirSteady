#pragma once

#include <QDialog>

namespace airsteady {

class ContactDialog : public QDialog {
  Q_OBJECT
 public:
  explicit ContactDialog(QWidget* parent = nullptr);
};

}  // namespace airsteady
