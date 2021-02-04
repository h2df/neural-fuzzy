#ifndef TRAINTHREAD_H
#define TRAINTHREAD_H
#include <QtCore>
#include "algorithm.h"


class TrainThread : public QThread
{
    Q_OBJECT
    NFTrainer* trainer;
    PendulumDataNormalizer *normalizer;
    std::string training_data_path;
    bool Initialize();
public:
    explicit TrainThread(QObject *parent, NFTrainer* trainer, PendulumDataNormalizer* normalizer, const std::string training_data_path);
    void run() override;
signals:
    void warning(std::string);
    void beyond_epoch_limit(unsigned);
    void train_nf(double, double);
    void train_success(double, double);
};

#endif // TRAINTHREAD_H
