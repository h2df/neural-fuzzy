#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H
#include <QtCore>
#include "algorithm.h"


class WorkerThread : public QThread
{
    Q_OBJECT
    NFTrainer* trainer;
    const TrainingDataParams _data_params;
public:
    explicit WorkerThread(QObject *parent, NFTrainer* trainer, const TrainingDataParams& training_data_params);
    void run() override;
signals:
    void warning(std::string);
    void beyond_epoch_limit(unsigned);
    void train_nf(double, double, unsigned);
    void train_success(double, double, unsigned, std::string);
};

#endif // WORKERTHREAD_H
