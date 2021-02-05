#include "testthread.h"

TestThread::TestThread(QObject *parent, NFSystem *nf, PendulumDataNormalizer *normalizer,const std::string test_data_path) :QThread(parent),
    nf(nf), normalizer(normalizer), test_data_path(test_data_path)
{
}

void TestThread::run(){
    std::ifstream test_f(test_data_path);
    if (!test_f.is_open()) {
        emit warning("Invalid test data path: " + test_data_path);
    }
    std::vector<NFDataSample> test_data;
    double pos, angle, output;
    while (test_f >> pos >> angle >> output) {
        test_data.push_back(NFDataSample{{pos, angle}, output});
    }
    test_f.close();
    test_data = normalizer->Normalize(test_data);
    NFTester tester(nf);
    double error = tester.CalcAvgError(test_data);
    if (error != error){
        emit warning("Some test data cannot be handled by the system... Try a better training strategy.");
        return;
    }
    emit(test_nf(error));
}

