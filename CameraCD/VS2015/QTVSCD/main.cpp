#include "qtvscd.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QTVSCD w;
    w.show();
    return a.exec();
}
