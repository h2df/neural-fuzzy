#ifndef TRAINTHREAD_H
#define TRAINTHREAD_H
#include <QtCore>
#include "algorithm.h"


class TrainThread : public QThread
{
    Q_OBJECT
    NFSystem *nf;
    const NFTrainParams params;
    PendulumDataNormalizer *normalizer;
    std::string training_data_path;
    bool Initialize();
public:
    TrainThread(QObject *parent, NFSystem *nf, const NFTrainParams params, PendulumDataNormalizer* normalizer, const std::string training_data_path);
    void run() override;
signals:
    void warning(std::string);
    void beyond_epoch_limit(unsigned);
    void train_nf(double, double, unsigned);
    void train_success(double, double, unsigned, std::string);
};

#endif // TRAINTHREAD_H
