#include <QApplication>
#include "main_window.hpp"

#include <glog/logging.h>

#include "common/file_utils.hpp"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_logbufsecs = 0;  // 立即刷出日志，方便抓崩溃点

  // TODO: Set log dir
  // std::string exe_folder;
  // GetExeFolder(&exe_folder);
  // const std::string log_folder = exe_folder + "/logs";
  // CreateFolder(log_folder);
  // FLAGS_log_dir = log_folder;

  QApplication app(argc, argv);

  airsteady::MainWindow w;
  app.installEventFilter(&w);
  w.show();

  return app.exec();
}
