#ifndef TESTTHREAD_H
#define TESTTHREAD_H
#include <QtCore>
#include "algorithm.h"


class TestThread : public QThread
{
    Q_OBJECT
    NFSystem *nf;
    std::string test_data_path;
public:
    explicit TestThread(QObject *parent, NFSystem *nf, std::string test_data_path);
    void run() override;
signals:
};

#endif // TESTTHREAD_H
