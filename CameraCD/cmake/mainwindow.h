#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>
#include <QTimer>
#include <QDebug>
#include <QString>
#include <QMainWindow>

#include <QThread>
#include <QQueue>
#include <QMutex>
#include <QWaitCondition>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define slots Q_SLOTS
#undef slots
#include "TorchScpt_class.h"

#include "yolo_class.hpp"
#include "ncnn_class.hpp"

// #include "infeRt_class.h"

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

enum class DetType
{
    None,
    darknet,
    RTint8,
    RTfp16,
    RTfp32,
    torchscript,
    tencentncnn,
    openvino
};
enum DataType
{
    Empty,
    image,
    video,
    camera
};
static QQueue<cv::Mat *> Main2Det;
static QQueue<cv::Mat *> Det2Main;

class Detthread : public QThread
{
    Q_OBJECT
public:
    Detthread();
    ~Detthread();

    std::atomic_bool pauseFlag;
    enum DetType mType;

public Q_SLOTS:
    void DoFromMain(int);

private:
    QMutex mutex;
    std::atomic_bool stopFlag;
    QWaitCondition condition;

    Detector *DarknetDetr;
    TorchScpt::Detector* TorchSptDetr;
    // TensorRT::Detector* RTint8Detr;
    // TensorRT::Detector* RTfp16Detr;
    // TensorRT::Detector* RTfp32Detr;
    YOLONCNN::Detector* NcnnDetr;
    // OpenVino::Detector* OpenvinoDetr;

protected:
    void run() override;
    void pause();
    void resume();
    void stop();
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
    void ExtraSR(); //Shade称为着色，把Render称为渲染
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    virtual bool eventFilter(QObject *obj, QEvent *event) override;
signals:
    void emit2det(int);

public Q_SLOTS:
    void FlushRL();
    void GenData();

    void DoBnImshow();
    void DoBnDetect();
    void DoBnStop_S();
    void DoBnStop_D();

    void DoActionImage();
    void DoActionVideo();
    void DoActionCamera();

    void DoActionOnnx();
    void DoActionFP32();
    void DoActionFP16();
    void DoActionINT8();
    void DoActionDarknet();
    void DoActionTorchSpt();

    void DoActionCopyRight();

private:
    enum DataType mInputType;
    void MsgBox(int);
    void ShowMessages(std::string);
    void DetMessages(std::string);

private:
    Ui::MainWindow *ui;
    QTimer *GenTimer, *FlushTimer;
    Detthread *mThread;

    cv::Mat Gen2Main;
    cv::VideoCapture cap;
    QString mPATH;
};
#endif
