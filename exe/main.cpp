#include <QApplication>
#include "main_window.hpp"

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);

  airsteady::MainWindow w;
  app.installEventFilter(&w);
  w.show();

  return app.exec();
}
