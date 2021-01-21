#include "workerthread.h"

WorkerThread::WorkerThread(QObject *parent, const NFTrainParams& params, const TrainingDataParams& training_data_params) :QThread(parent),
    _training_params(params), _data_params(training_data_params)
{
}

void WorkerThread::run(){
    auto data = initialize_data(_data_params);
    if (!data.valid) {
        emit warning("Invalid training data path: " + _data_params.training_data_path);
        return;
    }
    NFTrainer trainer = NFTrainer(_training_params);
    trainer.InitializeNNFromData(data);

    double error = 0;//1;
    double prevError = 0;

    int epoch=1;

    qDebug() << "============================================" ;
    qDebug() << "Training begins... ";
    do
    {
        if (trainer.ForceStopTraining()) {
            break;
        }

        trainer.TrainOneEpoch(data);
        prevError = error;
        error = trainer.CalcAvgError(data.training_data);
        qDebug() << "\n\t" << error ;
        qDebug() << "---<< Epoch # " << epoch << " >>---" << ", Error difference = " << fabs(error - prevError);


        //let user adjust learning rate manually to be consistent with other hyper parameters
        // if( fabs(prevError - error) < 0.000001){
        //     std::cout << "Error difference = " << fabs(error - prevError) << " \t\t\t\t--- adjusting learning rates... " <<  std::endl;
        //     trainer.AdjustLearningRates();
        // }

        epoch++;
        emit train_nf(error);

    } while (error > _training_params.error_threshold);
}
