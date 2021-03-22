#include "mainwindow.h"
#include "./ui_mainwindow.h"


#include <QMessageBox>
#include <QDebug>



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    det = new Detector("model/yolov3.cfg","model/yolov3.weights",0) ;

    connect(ui->actionpic,SIGNAL(triggered()),this,SLOT(messages()));
    connect(ui->actionyolov5,SIGNAL(triggered()),this,SLOT(messages()));
    

}

void MainWindow::messages()
{

    switch(types){
        case 1:
            QMessageBox::information(NULL, "Title", "Content",QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);break;
        case 2:
            QMessageBox::information(NULL, "Title", "Content",QMessageBox::Yes|QMessageBox::No); break;
        case 3:
            QMessageBox::information(NULL, "Title", "Content",QMessageBox::Yes|QMessageBox::No|
                             QMessageBox::Abort);break;
        case 4:
                {QMessageBox::StandardButton result= QMessageBox::information(NULL, "Title", "Content",QMessageBox::Yes|QMessageBox::No);
                switch (result)
                {
                case QMessageBox::Yes:
                    qDebug()<<"Yes";
                    break;
                case QMessageBox::No:
                    qDebug()<<"NO";
                    break;
                default:
                    break;
                }
                break;}
        case 5:
            QMessageBox::critical(NULL, "critical", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);break;
        case 6:
            QMessageBox::warning(NULL, "warning", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);break;
        case 7:
            QMessageBox::question(NULL, "question", "Content", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);break;
        case 8:
            QMessageBox::about(NULL, "About", "by hjwblog.com");break;
        default:
            break;    
    }
    ;
    showMessages(message);
}

void MainWindow::showMessages(string str){

    ui->statusbar->showMessage(tr("attention !"),0);
    
}
MainWindow::~MainWindow()
{
    delete ui;
}


WKthread::WKthread(){

}
void WKthread::run(){

    cv::Mat img;
    img = cv::imread("cat.jpg");
    cv::imshow("re",img);
    cv::imshow("",cv::imread("cat.jpg"));
}
