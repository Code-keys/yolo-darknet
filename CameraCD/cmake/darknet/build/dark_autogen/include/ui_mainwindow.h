/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionpic;
    QAction *actionvideo;
    QAction *actionfolder;
    QAction *actiontxt;
    QAction *actionexit;
    QAction *actiondet;
    QAction *actionyolov4;
    QAction *actionyolov5;
    QAction *actiontrr;
    QAction *actiontrrr;
    QAction *action1;
    QAction *action2;
    QAction *action3;
    QAction *action4;
    QAction *action1_2;
    QAction *action2_2;
    QAction *action3_2;
    QAction *action4_2;
    QAction *action5;
    QAction *action6;
    QAction *actionabout;
    QWidget *centralwidget;
    QLabel *label;
    QMenuBar *menubar;
    QMenu *menuinterface;
    QMenu *menudet;
    QMenu *menuplaceholder;
    QMenu *menuplaceholder1;
    QMenu *menuhelp;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(800, 600);
        actionpic = new QAction(MainWindow);
        actionpic->setObjectName(QStringLiteral("actionpic"));
        actionvideo = new QAction(MainWindow);
        actionvideo->setObjectName(QStringLiteral("actionvideo"));
        actionfolder = new QAction(MainWindow);
        actionfolder->setObjectName(QStringLiteral("actionfolder"));
        actiontxt = new QAction(MainWindow);
        actiontxt->setObjectName(QStringLiteral("actiontxt"));
        actionexit = new QAction(MainWindow);
        actionexit->setObjectName(QStringLiteral("actionexit"));
        actiondet = new QAction(MainWindow);
        actiondet->setObjectName(QStringLiteral("actiondet"));
        actionyolov4 = new QAction(MainWindow);
        actionyolov4->setObjectName(QStringLiteral("actionyolov4"));
        actionyolov5 = new QAction(MainWindow);
        actionyolov5->setObjectName(QStringLiteral("actionyolov5"));
        actiontrr = new QAction(MainWindow);
        actiontrr->setObjectName(QStringLiteral("actiontrr"));
        actiontrrr = new QAction(MainWindow);
        actiontrrr->setObjectName(QStringLiteral("actiontrrr"));
        action1 = new QAction(MainWindow);
        action1->setObjectName(QStringLiteral("action1"));
        action2 = new QAction(MainWindow);
        action2->setObjectName(QStringLiteral("action2"));
        action3 = new QAction(MainWindow);
        action3->setObjectName(QStringLiteral("action3"));
        action4 = new QAction(MainWindow);
        action4->setObjectName(QStringLiteral("action4"));
        action1_2 = new QAction(MainWindow);
        action1_2->setObjectName(QStringLiteral("action1_2"));
        action2_2 = new QAction(MainWindow);
        action2_2->setObjectName(QStringLiteral("action2_2"));
        action3_2 = new QAction(MainWindow);
        action3_2->setObjectName(QStringLiteral("action3_2"));
        action4_2 = new QAction(MainWindow);
        action4_2->setObjectName(QStringLiteral("action4_2"));
        action5 = new QAction(MainWindow);
        action5->setObjectName(QStringLiteral("action5"));
        action6 = new QAction(MainWindow);
        action6->setObjectName(QStringLiteral("action6"));
        actionabout = new QAction(MainWindow);
        actionabout->setObjectName(QStringLiteral("actionabout"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        label = new QLabel(centralwidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(0, 0, 801, 551));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 22));
        menuinterface = new QMenu(menubar);
        menuinterface->setObjectName(QStringLiteral("menuinterface"));
        menudet = new QMenu(menubar);
        menudet->setObjectName(QStringLiteral("menudet"));
        menuplaceholder = new QMenu(menubar);
        menuplaceholder->setObjectName(QStringLiteral("menuplaceholder"));
        menuplaceholder1 = new QMenu(menubar);
        menuplaceholder1->setObjectName(QStringLiteral("menuplaceholder1"));
        menuhelp = new QMenu(menubar);
        menuhelp->setObjectName(QStringLiteral("menuhelp"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuinterface->menuAction());
        menubar->addAction(menudet->menuAction());
        menubar->addAction(menuplaceholder->menuAction());
        menubar->addAction(menuplaceholder1->menuAction());
        menubar->addAction(menuhelp->menuAction());
        menuinterface->addAction(actionpic);
        menuinterface->addAction(actionvideo);
        menuinterface->addAction(actionfolder);
        menuinterface->addAction(actiontxt);
        menudet->addAction(actiondet);
        menudet->addAction(actionyolov4);
        menudet->addAction(actionyolov5);
        menuplaceholder->addAction(action1);
        menuplaceholder->addAction(action2);
        menuplaceholder->addAction(action3);
        menuplaceholder->addAction(action4);
        menuplaceholder1->addAction(action1_2);
        menuplaceholder1->addAction(action2_2);
        menuplaceholder1->addAction(action3_2);
        menuplaceholder1->addAction(action4_2);
        menuplaceholder1->addAction(action5);
        menuplaceholder1->addAction(action6);
        menuhelp->addAction(actionabout);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        actionpic->setText(QApplication::translate("MainWindow", "pic", Q_NULLPTR));
        actionvideo->setText(QApplication::translate("MainWindow", "video", Q_NULLPTR));
        actionfolder->setText(QApplication::translate("MainWindow", "folder", Q_NULLPTR));
        actiontxt->setText(QApplication::translate("MainWindow", "txt", Q_NULLPTR));
        actionexit->setText(QApplication::translate("MainWindow", "exit", Q_NULLPTR));
        actiondet->setText(QApplication::translate("MainWindow", "yolov3", Q_NULLPTR));
        actionyolov4->setText(QApplication::translate("MainWindow", "yolov4", Q_NULLPTR));
        actionyolov5->setText(QApplication::translate("MainWindow", "yolov5", Q_NULLPTR));
        actiontrr->setText(QApplication::translate("MainWindow", "trr", Q_NULLPTR));
        actiontrrr->setText(QApplication::translate("MainWindow", "trrr", Q_NULLPTR));
        action1->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        action2->setText(QApplication::translate("MainWindow", "2", Q_NULLPTR));
        action3->setText(QApplication::translate("MainWindow", "3", Q_NULLPTR));
        action4->setText(QApplication::translate("MainWindow", "4", Q_NULLPTR));
        action1_2->setText(QApplication::translate("MainWindow", "1", Q_NULLPTR));
        action2_2->setText(QApplication::translate("MainWindow", "2", Q_NULLPTR));
        action3_2->setText(QApplication::translate("MainWindow", "3", Q_NULLPTR));
        action4_2->setText(QApplication::translate("MainWindow", "4", Q_NULLPTR));
        action5->setText(QApplication::translate("MainWindow", "5", Q_NULLPTR));
        action6->setText(QApplication::translate("MainWindow", "6", Q_NULLPTR));
        actionabout->setText(QApplication::translate("MainWindow", "about", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "PIC", Q_NULLPTR));
        menuinterface->setTitle(QApplication::translate("MainWindow", "interface", Q_NULLPTR));
        menudet->setTitle(QApplication::translate("MainWindow", "det", Q_NULLPTR));
        menuplaceholder->setTitle(QApplication::translate("MainWindow", "placeholder", Q_NULLPTR));
        menuplaceholder1->setTitle(QApplication::translate("MainWindow", "placeholder1", Q_NULLPTR));
        menuhelp->setTitle(QApplication::translate("MainWindow", "help", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
