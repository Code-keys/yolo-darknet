/********************************************************************************
** Form generated from reading UI file 'qtvscd.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTVSCD_H
#define UI_QTVSCD_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *action_darknet;
    QAction *action_other;
    QAction *action_Exit_Q;
    QAction *action_pic;
    QAction *action_video;
    QAction *action_txt;
    QAction *action_folder;
    QAction *action_camera;
    QAction *action_Reset;
    QAction *action_yolov3_rt;
    QAction *action_yolov4_rt;
    QAction *action_yolov5_rt;
    QAction *actioncopr_right;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QPushButton *BN_4;
    QSpacerItem *horizontalSpacer_6;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *BN_2;
    QPushButton *BN_3;
    QLabel *label_R;
    QLabel *label_L;
    QPushButton *BN_1;
    QMenuBar *menuBar;
    QMenu *menufile_F;
    QMenu *menutensorRT;
    QMenu *menuhelp;
    QMenu *menuhelp_2;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1304, 611);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMouseTracking(true);
        MainWindow->setToolTipDuration(0);
        action_darknet = new QAction(MainWindow);
        action_darknet->setObjectName(QStringLiteral("action_darknet"));
        action_darknet->setCheckable(true);
        action_darknet->setChecked(true);
        action_darknet->setIconVisibleInMenu(true);
        action_other = new QAction(MainWindow);
        action_other->setObjectName(QStringLiteral("action_other"));
        action_other->setCheckable(true);
        action_Exit_Q = new QAction(MainWindow);
        action_Exit_Q->setObjectName(QStringLiteral("action_Exit_Q"));
        action_pic = new QAction(MainWindow);
        action_pic->setObjectName(QStringLiteral("action_pic"));
        action_pic->setCheckable(true);
        action_pic->setChecked(false);
        action_pic->setEnabled(true);
        action_pic->setIconVisibleInMenu(false);
        action_video = new QAction(MainWindow);
        action_video->setObjectName(QStringLiteral("action_video"));
        action_video->setCheckable(true);
        action_txt = new QAction(MainWindow);
        action_txt->setObjectName(QStringLiteral("action_txt"));
        action_txt->setCheckable(true);
        action_folder = new QAction(MainWindow);
        action_folder->setObjectName(QStringLiteral("action_folder"));
        action_folder->setCheckable(true);
        action_camera = new QAction(MainWindow);
        action_camera->setObjectName(QStringLiteral("action_camera"));
        action_camera->setCheckable(true);
        action_camera->setChecked(true);
        action_Reset = new QAction(MainWindow);
        action_Reset->setObjectName(QStringLiteral("action_Reset"));
        action_yolov3_rt = new QAction(MainWindow);
        action_yolov3_rt->setObjectName(QStringLiteral("action_yolov3_rt"));
        action_yolov3_rt->setCheckable(true);
        action_yolov4_rt = new QAction(MainWindow);
        action_yolov4_rt->setObjectName(QStringLiteral("action_yolov4_rt"));
        action_yolov4_rt->setCheckable(true);
        action_yolov5_rt = new QAction(MainWindow);
        action_yolov5_rt->setObjectName(QStringLiteral("action_yolov5_rt"));
        action_yolov5_rt->setCheckable(true);
        actioncopr_right = new QAction(MainWindow);
        actioncopr_right->setObjectName(QStringLiteral("actioncopr_right"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        BN_4 = new QPushButton(centralWidget);
        BN_4->setObjectName(QStringLiteral("BN_4"));
        sizePolicy.setHeightForWidth(BN_4->sizePolicy().hasHeightForWidth());
        BN_4->setSizePolicy(sizePolicy);

        gridLayout->addWidget(BN_4, 1, 5, 1, 1);

        horizontalSpacer_6 = new QSpacerItem(166, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_6, 1, 1, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(167, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_5, 1, 6, 1, 1);

        BN_2 = new QPushButton(centralWidget);
        BN_2->setObjectName(QStringLiteral("BN_2"));
        sizePolicy.setHeightForWidth(BN_2->sizePolicy().hasHeightForWidth());
        BN_2->setSizePolicy(sizePolicy);

        gridLayout->addWidget(BN_2, 1, 3, 1, 1);

        BN_3 = new QPushButton(centralWidget);
        BN_3->setObjectName(QStringLiteral("BN_3"));
        sizePolicy.setHeightForWidth(BN_3->sizePolicy().hasHeightForWidth());
        BN_3->setSizePolicy(sizePolicy);

        gridLayout->addWidget(BN_3, 1, 4, 1, 1);

        label_R = new QLabel(centralWidget);
        label_R->setObjectName(QStringLiteral("label_R"));
        label_R->setEnabled(true);
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_R->sizePolicy().hasHeightForWidth());
        label_R->setSizePolicy(sizePolicy1);
        label_R->setMinimumSize(QSize(640, 512));

        gridLayout->addWidget(label_R, 2, 4, 1, 3);

        label_L = new QLabel(centralWidget);
        label_L->setObjectName(QStringLiteral("label_L"));
        sizePolicy1.setHeightForWidth(label_L->sizePolicy().hasHeightForWidth());
        label_L->setSizePolicy(sizePolicy1);
        label_L->setMinimumSize(QSize(640, 512));
        label_L->setTextFormat(Qt::AutoText);

        gridLayout->addWidget(label_L, 2, 0, 1, 4);

        BN_1 = new QPushButton(centralWidget);
        BN_1->setObjectName(QStringLiteral("BN_1"));
        sizePolicy.setHeightForWidth(BN_1->sizePolicy().hasHeightForWidth());
        BN_1->setSizePolicy(sizePolicy);

        gridLayout->addWidget(BN_1, 1, 2, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setEnabled(true);
        menuBar->setGeometry(QRect(0, 0, 1304, 28));
        menufile_F = new QMenu(menuBar);
        menufile_F->setObjectName(QStringLiteral("menufile_F"));
        menufile_F->setGeometry(QRect(306, 147, 160, 175));
        menutensorRT = new QMenu(menufile_F);
        menutensorRT->setObjectName(QStringLiteral("menutensorRT"));
        menuhelp = new QMenu(menuBar);
        menuhelp->setObjectName(QStringLiteral("menuhelp"));
        menuhelp->setGeometry(QRect(386, 147, 158, 175));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(menuhelp->sizePolicy().hasHeightForWidth());
        menuhelp->setSizePolicy(sizePolicy2);
        menuhelp->setAutoFillBackground(false);
        menuhelp_2 = new QMenu(menuBar);
        menuhelp_2->setObjectName(QStringLiteral("menuhelp_2"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        statusBar->setEnabled(true);
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(statusBar->sizePolicy().hasHeightForWidth());
        statusBar->setSizePolicy(sizePolicy3);
        QPalette palette;
        QBrush brush(QColor(173, 127, 168, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush);
        QBrush brush1(QColor(190, 190, 190, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush1);
        statusBar->setPalette(palette);
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menufile_F->menuAction());
        menuBar->addAction(menuhelp->menuAction());
        menuBar->addAction(menuhelp_2->menuAction());
        menufile_F->addAction(action_darknet);
        menufile_F->addAction(menutensorRT->menuAction());
        menufile_F->addAction(action_other);
        menufile_F->addAction(action_Reset);
        menufile_F->addAction(action_Exit_Q);
        menutensorRT->addAction(action_yolov3_rt);
        menutensorRT->addAction(action_yolov4_rt);
        menutensorRT->addAction(action_yolov5_rt);
        menuhelp->addAction(action_pic);
        menuhelp->addAction(action_video);
        menuhelp->addAction(action_txt);
        menuhelp->addAction(action_folder);
        menuhelp->addAction(action_camera);
        menuhelp_2->addAction(actioncopr_right);

        retranslateUi(MainWindow);
        QObject::connect(action_Exit_Q, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        action_darknet->setText(QApplication::translate("MainWindow", "Darknet", Q_NULLPTR));
        action_other->setText(QApplication::translate("MainWindow", "Other", Q_NULLPTR));
        action_Exit_Q->setText(QApplication::translate("MainWindow", "Exit(Q)", Q_NULLPTR));
        action_pic->setText(QApplication::translate("MainWindow", "Pic", Q_NULLPTR));
        action_video->setText(QApplication::translate("MainWindow", "Video", Q_NULLPTR));
        action_txt->setText(QApplication::translate("MainWindow", "Txt", Q_NULLPTR));
        action_folder->setText(QApplication::translate("MainWindow", "Folder", Q_NULLPTR));
        action_camera->setText(QApplication::translate("MainWindow", "Camera", Q_NULLPTR));
        action_Reset->setText(QApplication::translate("MainWindow", "Reset", Q_NULLPTR));
        action_yolov3_rt->setText(QApplication::translate("MainWindow", "YoloV3", Q_NULLPTR));
        action_yolov4_rt->setText(QApplication::translate("MainWindow", "YoloV4", Q_NULLPTR));
        action_yolov5_rt->setText(QApplication::translate("MainWindow", "YoloV5", Q_NULLPTR));
        actioncopr_right->setText(QApplication::translate("MainWindow", "Copy@Right", Q_NULLPTR));
        BN_4->setText(QApplication::translate("MainWindow", "stop", Q_NULLPTR));
        BN_2->setText(QApplication::translate("MainWindow", "Imshow", Q_NULLPTR));
        BN_3->setText(QApplication::translate("MainWindow", "Detect", Q_NULLPTR));
        label_R->setText(QApplication::translate("MainWindow", "DETECTION", Q_NULLPTR));
        label_L->setText(QApplication::translate("MainWindow", "VIEW", Q_NULLPTR));
        BN_1->setText(QApplication::translate("MainWindow", "stop", Q_NULLPTR));
        menufile_F->setTitle(QApplication::translate("MainWindow", "Model(M)", Q_NULLPTR));
        menutensorRT->setTitle(QApplication::translate("MainWindow", "TensorRT", Q_NULLPTR));
        menuhelp->setTitle(QApplication::translate("MainWindow", "Source(S)", Q_NULLPTR));
        menuhelp_2->setTitle(QApplication::translate("MainWindow", "Help", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTVSCD_H
