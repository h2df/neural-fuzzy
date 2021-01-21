#include "nfdash.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    NFDash w;
    w.show();
    return a.exec();
}
