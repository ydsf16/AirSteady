#include <QApplication>
#include "main_window.hpp"

#include <glog/logging.h>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_logbufsecs = 0;  // 立即刷出日志，方便抓崩溃点

  QApplication app(argc, argv);

  airsteady::MainWindow w;
  app.installEventFilter(&w);
  w.show();

  return app.exec();
}
