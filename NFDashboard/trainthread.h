#ifndef TRAINTHREAD_H
#define TRAINTHREAD_H
#include <QtCore>
#include "algorithm.h"


class TrainThread : public QThread
{
    Q_OBJECT
    NFTrainer* trainer;
    const TrainingDataParams _data_params;
public:
    explicit TrainThread(QObject *parent, NFTrainer* trainer, const TrainingDataParams& training_data_params);
    void run() override;
signals:
    void warning(std::string);
    void beyond_epoch_limit(unsigned);
    void train_nf(double, double);
    void train_success(double, double);
};

#endif // TRAINTHREAD_H
