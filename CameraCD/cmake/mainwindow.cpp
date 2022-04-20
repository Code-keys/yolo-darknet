#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QString>
#include <QDebug>
#include <QImage>
#include <QtGui>
#include <QAction>

/* *******************************         this       is        det         thread       ******************************** */

Detthread::Detthread() : pauseFlag(true), mType(DetType::None), stopFlag(false)
{
    DarknetDetr = new Detector(
        "../models/yolov5sm.cfg",
        "../models/yolov5sm.weights", 0);
    TorchSptDetr = new  TorchScpt::Detector("../models/yolov5sm.torchscript", "", -1);
    NcnnDetr = new YOLONCNN::Detector( "../models/yolov5sm.param", "../models/yolov5sm.bin", {608, 608}, -1);
    // TensorRT::Detector* RTint8Detr = nullptr;
    // TensorRT::Detector* RTfp16Detr = nullptr;
    // TensorRT::Detector* RTfp32Detr = nullptr;
    // OnnX::Detector* OnnxDetr = nullptr;
    // OpenVino::Detector* OpenvinoDetr = nullptr;
};

void Detthread::DoFromMain(int i)
{
    if (i == -2)
    {
        stop();
    }
    else if (i == -1)
    {
        pause();
    }
    else if (i == 0)
    {
        resume();
    }
    else if (i == 1)
    {
        mType = DetType::darknet;
    }
    else if (i == 2)
    {
        mType = DetType::RTint8;
    }
    else if (i == 3)
    {
        mType = DetType::RTfp16;
    }
    else if (i == 4)
    {
        mType = DetType::RTfp32;
    }
    else if (i == 5)
    {
        mType = DetType::torchscript;
    }
    else if (i == 6)
    {
        mType = DetType::tencentncnn;
    }
    else if (i == 7)
    {
        mType = DetType::openvino;
    }
};

void Detthread::stop()
{
    if (QThread::isRunning())
    {
        stopFlag = true;
    }
};

void Detthread::pause()
{
    if (QThread::isRunning())
    {
        pauseFlag = true;
    }
};

void Detthread::resume()
{
    if (QThread::isRunning())
    {
        pauseFlag = false;
        condition.wakeAll();
    }
};

void Detthread::run()
{
    while (1)
    {
        if (stopFlag)
            return;
        if (pauseFlag)
        {
            mutex.lock();
            condition.wait(&mutex);
            mutex.unlock();
        }

        if (Main2Det.isEmpty())
            continue;
        qDebug() << "Main2Det imgs remain :" << Main2Det.size();
        while (Main2Det.size() > 1)
        {
            cv::Mat *img = Main2Det.dequeue();
            delete img;
        };

        cv::Mat *img = Main2Det.dequeue();
        float t = 0;
        if (mType == DetType::darknet)
            t = DarknetDetr->predict_cv(*img, 0.2, 0.4);
        if( mType == DetType::torchscript ) t =  TorchSptDetr->predict_cv(*img, 0.2, 0.4);
        // if( mType == DetType::RTint8 ) t =  RTint8Detr->predict_cv(*img, 0.2, 0.4);
        // if( mType == DetType::RTfp16 ) t =  RTfp16Detr->predict_cv(*img, 0.2, 0.4);
        // if( mType == DetType::RTfp32 ) t =  RTfp32Detr->predict_cv(*img, 0.2, 0.4);
        if( mType == DetType::tencentncnn ) t =  NcnnDetr->predict_cv(*img, 0.2, 0.4);
        // if( mType == DetType::openvino ) t =  OpenvinoDetr->predict_cv(*img, 0.2, 0.4);

        Det2Main.enqueue(img);
    }
};

Detthread::~Detthread()
{
    delete DarknetDetr;
    // delete TorchSptDetr;
};

/* ********************************         this       is        main         thread       ******************************** */

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    GenTimer = new QTimer(this);   // for genData
    FlushTimer = new QTimer(this); // for flash  60 Fps
    connect(GenTimer, SIGNAL(timeout()), this, SLOT(GenData()));
    connect(FlushTimer, SIGNAL(timeout()), this, SLOT(FlushRL()));

    mInputType = Empty;
    mPATH = "";

    mThread = new Detthread();
    mThread->start();
    connect(this, SIGNAL(emit2det(int)), mThread, SLOT(DoFromMain(int)));

    this->ExtraSR();
};

void MainWindow::ExtraSR()
{
    ui->Imshow->setStyleSheet("QPushButton{border-image: url(:/src/btn1.1.png);}"
                              "QPushButton:hover{border-image: url(:/src/btn1.2.png);}"
                              "QPushButton:pressed{border-image: url(:/src/btn1.3.png);}");
    ui->Detect->setStyleSheet("QPushButton{border-image: url(:/src/btn1.1.png);}"
                              "QPushButton:hover{border-image: url(:/src/btn1.2.png);}"
                              "QPushButton:pressed{border-image: url(:/src/btn1.3.png);}");
    ui->Stop_S->setStyleSheet("QPushButton{border-image: url(:/src/btn2.0.png);}"
                              "QPushButton:hover{border-image: url(:/src/btn2.1.png);}"
                              "QPushButton:pressed{border-image: url(:/src/btn2.2.png);}");
    ui->Stop_D->setStyleSheet("QPushButton{border-image: url(:/src/btn2.0.png);}"
                              "QPushButton:hover{border-image: url(:/src/btn2.1.png);}"
                              "QPushButton:pressed{border-image: url(:/src/btn2.2.png);}");
    connect(ui->Imshow, SIGNAL(clicked()), this, SLOT(DoBnImshow()));
    connect(ui->Detect, SIGNAL(clicked()), this, SLOT(DoBnDetect()));
    connect(ui->Stop_S, SIGNAL(clicked()), this, SLOT(DoBnStop_S()));
    connect(ui->Stop_D, SIGNAL(clicked()), this, SLOT(DoBnStop_D()));

    ui->labelOrg->setStyleSheet("QLabel{border-image: url(:/src/btn1.png);}");
    ui->labelDet->setStyleSheet("QLabel{border-image: url(:/src/btn1.png);}");
    ui->labelOrg->installEventFilter(this);
    ui->labelDet->installEventFilter(this);

    connect(ui->actionImage, SIGNAL(triggered()), this, SLOT(DoActionImage()));
    connect(ui->actionVideo, SIGNAL(triggered()), this, SLOT(DoActionVideo()));
    connect(ui->actionCamera, SIGNAL(triggered()), this, SLOT(DoActionCamera()));
    connect(ui->actionFolder, SIGNAL(triggered()), this, SLOT(DoActionImage()));
    connect(ui->actionTxT, SIGNAL(triggered()), this, SLOT(DoActionImage()));
    ui->actionImage->setShortcut(QKeySequence( Qt::Key_Q )); 
    ui->actionVideo->setShortcut(QKeySequence( Qt::Key_W )); 
    ui->actionCamera->setShortcut(QKeySequence( Qt::Key_E )); 
    ui->actionFolder->setShortcut(QKeySequence( Qt::Key_R )); 
    ui->actionTxT->setShortcut(QKeySequence( Qt::Key_T )); 

    connect(ui->actionDarknet, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));
    connect(ui->actionFP32, SIGNAL(triggered()), this, SLOT(DoActionFP32()));
    connect(ui->actionFP16, SIGNAL(triggered()), this, SLOT(DoActionFP16()));
    connect(ui->actionINT8, SIGNAL(triggered()), this, SLOT(DoActionINT8()));
    connect(ui->actionTorchSpt, SIGNAL(triggered()), this, SLOT(DoActionTorchSpt()));
    connect(ui->actionOnnx, SIGNAL(triggered()), this, SLOT(DoActionOnnx()));
    ui->actionDarknet->setShortcut(QKeySequence( Qt::Key_A )); 
    ui->actionFP32->setShortcut(QKeySequence( Qt::Key_S )); 
    ui->actionFP16->setShortcut(QKeySequence( Qt::Key_D )); 
    ui->actionINT8->setShortcut(QKeySequence( Qt::Key_F )); 
    ui->actionTorchSpt->setShortcut(QKeySequence( Qt::Key_G )); 
    ui->actionOnnx->setShortcut(QKeySequence( Qt::Key_H )); 

    connect(ui->action2NoSave, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));
    connect(ui->action2Img, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));
    connect(ui->action2Json, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));
    connect(ui->action2TxT, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));

    connect(ui->actionCopyRight, SIGNAL(triggered()), this, SLOT(DoActionCopyRight()));
    connect(ui->actionExitQ, SIGNAL(triggered()), this, SLOT(DoActionDarknet()));
    ui->actionCopyRight->setShortcut(QKeySequence( Qt::Key_Comma + Qt::Key_I )); 
    ui->actionExitQ->setShortcut(QKeySequence( Qt::Key_Escape )); 
};

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    static int i;
    if (obj == ui->labelOrg)
    {
        if (event->type() == QEvent::MouseButtonPress) // double click enent
        {
            i++;
            if (i % 2 == 0) 
            {
                ui->labelOrg->setWindowFlags(Qt::Dialog);
                ui->labelOrg->showFullScreen(); 
            }
            else
            {
                ui->labelOrg->setWindowFlags(Qt::SubWindow);
                ui->labelOrg->showNormal(); 
            };
        }
        return QObject::eventFilter(obj, event);
    }
    else if (obj == ui->labelDet)
    {
        if (event->type() == QEvent::MouseButtonPress) 
        {
            i++;
            if (i % 2 == 0) 
            {
                ui->labelDet->setWindowFlags(Qt::Dialog);
                ui->labelDet->showFullScreen(); 
            }
            else
            {
                ui->labelDet->setWindowFlags(Qt::SubWindow);
                ui->labelDet->showNormal(); 
            };
        }
        return QObject::eventFilter(obj, event);
    }
    //    return QObject::eventFilter(obj, event);
};

void MainWindow::GenData()
{ // lock();
    switch (mInputType)
    {
    case Empty:
        break;
    case image:
    {
        cv::Mat temp = cv::imread(mPATH.toStdString());
        Gen2Main = temp.clone();
        auto t = new cv::Mat(temp);
        if (!mThread->pauseFlag)
            Main2Det.enqueue(t);
        break;
    }
    case video:
    {
        cv::Mat temp;
        cap >> temp;
        if (temp.empty() || !cap.isOpened())
        {
            cap.open(mPATH.toStdString());
        }
        else
        {
            auto t = new cv::Mat(temp);
            Gen2Main = temp.clone();
            if (!mThread->pauseFlag)
                Main2Det.enqueue(t);
        }
        break;
    }
    case camera:
    {
        cv::Mat temp;
        cap >> temp;
        if (temp.empty() || !cap.isOpened())
        {
            cap.open(0);
        }
        else
        {
            auto t = new cv::Mat(temp);
            Gen2Main = temp.clone();
            if (!mThread->pauseFlag)
                Main2Det.enqueue(t);
        }
        break;
    }
    } // unlock(); case end
};

void MainWindow::FlushRL()
{
    // lock();
    if (!Det2Main.isEmpty())
    {
        cv::Mat *deted = Det2Main.dequeue();
        //cv::Mat Image(const unsigned char*)deted->data);
        //cv::cvtColor(Image, Image, 0); // COLOR_BGR2RGB);
        QImage disImage = QImage((const unsigned char *)deted->data,
                                 deted->cols, deted->rows, QImage::Format_RGB888);       // 888是 three channel
        ui->labelDet->setPixmap(QPixmap::fromImage(disImage.scaled(ui->labelDet->size(),  
                                                                   Qt::KeepAspectRatio  )));
        delete deted;
    }
    else
    {
        // this->ShowMessage( " No images Detected !");
    }
    // unlock();
    if (!Gen2Main.empty())
    {
        // cv::cvtColor(Gen2Main, Gen2Main, 0); // COLOR_BGR2RGB);
        QImage disImage = QImage((const unsigned char *)Gen2Main.data,
                                 Gen2Main.cols, Gen2Main.rows, QImage::Format_RGB888);  
        ui->labelOrg->setPixmap(QPixmap::fromImage(disImage.scaled(ui->labelOrg->size(), 
                                                                   Qt::KeepAspectRatio )));
        Gen2Main.release();
    }
    else
    {
        // this->DetMessage( " No images Produced !");
    }
    // unlock();
};

void MainWindow::DoBnImshow()
{
    if (mPATH == "")
        QMessageBox::information(NULL, "Warning", "Please select Input !", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    else
    {
        GenTimer->start(51);
        FlushTimer->start(29);
    }
    qDebug() << "Imshow Clicked !";
};

void MainWindow::DoBnStop_S()
{
    emit emit2det(-1);
    GenTimer->stop();
    FlushTimer->stop();
};

void MainWindow::DoActionImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Images"), "..", tr("Image File(*.jpg| *.jpeg| *.png)"));
    if (fileName.isEmpty())
        return;

    auto i = cv::imread(fileName.toStdString());
    if (i.empty())
        return this->ShowMessages("Img Read Error!");
    this->ShowMessages("Image path is Ready!");
    mPATH = fileName;
    mInputType = image;
};

void MainWindow::DoActionVideo()
{
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this, tr("Open Images"), "..", tr("Image File(*.mp4| *.avi | *.mov)"));
    if (fileName.isEmpty())
        return;

    cap.open(fileName.toStdString());
    if (cap.isOpened())
    {
        this->ShowMessages("VideoCap is Ready!");
        mPATH = fileName;
        mInputType = video;
    }
};

void MainWindow::DoActionCamera()
{
    cap.open(0);
    if (cap.isOpened())
    {
        cv::Mat tmp;
        cap >> tmp;
        if (tmp.empty())
            return;
        this->ShowMessages("Camera is Ready!");
        mPATH = "0";
        mInputType = camera;
    }
};

void MainWindow::DoBnDetect()
{
    if (mPATH == "")
        QMessageBox::information(NULL, "Warning", "No Input For Det !", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    else
    {
        if (mThread->mType == DetType::None)
        {
            emit emit2det(1);
            QMessageBox::information(NULL, "Warning", "Using Darknet as default!", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        }
        emit emit2det(0);
    }
};
void MainWindow::DoBnStop_D()
{
    emit emit2det(-1);
};
void MainWindow::DoActionDarknet()
{
    emit emit2det(1);
};
void MainWindow::DoActionINT8()
{
    emit emit2det(2);
};
void MainWindow::DoActionFP16()
{
    emit emit2det(3);
};
void MainWindow::DoActionFP32()
{
    emit emit2det(4);
};
void MainWindow::DoActionTorchSpt()
{
    emit emit2det(5);
};
void MainWindow::DoActionOnnx()
{
    emit emit2det(6);
};

void MainWindow::MsgBox(int types)
{
    switch (types)
    {
    case 1:
        QMessageBox::information(NULL, "Title", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        break;
    case 2:
        QMessageBox::information(NULL, "Title", "Content", QMessageBox::Yes | QMessageBox::No);
        break;
    case 3:
        QMessageBox::information(NULL, "Title", "Content", QMessageBox::Yes | QMessageBox::No | QMessageBox::Abort);
        break;
    case 4:
    {
        QMessageBox::StandardButton result = QMessageBox::question(NULL, "Title", "Content", QMessageBox::Yes | QMessageBox::No);
        switch (result)
        {
        case QMessageBox::Yes:
            qDebug() << "Yes";
            break;
        case QMessageBox::No:
            qDebug() << "NO";
            break;
        default:
            break;
        }
        break;
    }
    case 5:
        QMessageBox::critical(NULL, "critical", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        break;
    case 6:
        QMessageBox::warning(NULL, "warning", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        break;
    case 7:
        QMessageBox::question(NULL, "question", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        break;
    case 8:
        QMessageBox::about(NULL, "About", "by xiaoxiaochenxu.com");
        break;
    default:
        break;
    };
    ShowMessages("cao");
};
void MainWindow::ShowMessages(std::string str)
{
    ui->ShowMsg->setText(QString::fromStdString(str));
};
void MainWindow::DetMessages(std::string str)
{
    ui->DetMsg->setText(QString::fromStdString(str));
};
void MainWindow::DoActionCopyRight()
{
    QMessageBox::about(NULL, "About", "Suport By xiaoxiaochenxu.top");
};

MainWindow::~MainWindow()
{
    emit emit2det(-2);
    delete ui;
};
