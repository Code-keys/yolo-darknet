#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <string>
using namespace std;

#include "yolo_v2_class.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { 
    class WKthread;
    class MainWindow;
 }
QT_END_NAMESPACE

class WKthread:public QThread
{
    Q_OBJECT
public:
    WKthread();
protected:
    void run(); // 新线程入口
// 省略掉一些内容
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
    // QAction *actionpic;
    // QAction *actionvideo;
    // QAction *actionfolder;
    // QAction *actiontxt;
    // QAction *actionexit;
    // QAction *actiondet;
    // QAction *actionyolov4;
    // QAction *actionyolov5;
    // QAction *actiontrr;
    // QAction *actiontrrr;
    // QAction *action1;
    // QAction *action2;
    // QAction *action3;
    // QAction *action4;
    // QAction *action1_2;
    // QAction *action2_2;
    // QAction *action3_2;
    // QAction *action4_2;
    // QAction *action5;
    // QAction *action6;
    // QAction *actionabout;
    // QWidget *centralwidget;
    // QLabel *label;
    // QMenuBar *menubar;
    // QMenu *menuinterface;
    // QMenu *menudet;
    // QMenu *menuplaceholder;
    // QMenu *menuplaceholder1;
    // QMenu *menuhelp;
    // QStatusBar *statusbar;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    // emit();

public slots:
    void messages();
    void showMessages(string str);

private:
    Ui::MainWindow *ui;
    WKthread* m_thread;
    Detector* det = nullptr;

    cv::Mat* img;
    int types{1};
    string message{"attention !"};



};




#endif // MAINWINDOW_H
