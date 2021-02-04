#include "testthread.h"

TestThread::TestThread(QObject *parent, NFSystem *nf, const std::string test_data_path) :QThread(parent),
    nf(nf), test_data_path(test_data_path)
{
}

void TestThread::run(){
    auto i = nf->GetRulesReport();
    auto j = test_data_path;
    return;
}
