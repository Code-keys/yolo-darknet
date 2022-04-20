#include "mainwindow.h"
#include <QApplication>
#include <QMessageBox>


int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    MainWindow w;

    QMessageBox::StandardButton _main_QMBOX_ = QMessageBox::information(NULL, " Welcome", "Join or Quit ? ", QMessageBox::Yes);
    if (_main_QMBOX_ == QMessageBox::Yes)
    {                    // 利用Accepted返回值判断按钮是否被按下
        w.show();        // 如果被按下，显示主窗口
        return a.exec(); // 程序一直执行，直到主窗口关闭
    }
}
