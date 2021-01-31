#include "workerthread.h"

WorkerThread::WorkerThread(QObject *parent, const NFTrainParams& params, const TrainingDataParams& training_data_params) :QThread(parent),
    _training_params(params), _data_params(training_data_params)
{
}

void WorkerThread::run(){
    NFTrainer trainer = NFTrainer(_training_params);
    trainer.Initialize(_data_params);
    if (!trainer.HasTrainingDataReady()) {
        emit warning("Invalid training data path: " + _data_params.training_data_path);
        return;
    }

    double error = 0;//1;
    double prevError = 0;

    qDebug() << "============================================" ;
    qDebug() << "Training begins... ";
    do
    {
        if (trainer.ForceStopTraining()) {
            emit beyond_epoch_limit(trainer.epoch_count);
            break;
        }

        trainer.TrainOneEpoch();
        prevError = error;
        error = trainer.CalcValidationError();
        qDebug() << "\n\t" << error ;
        qDebug() << "---<< Epoch # " << trainer.epoch_count << " >>---" << ", Error difference = " << fabs(error - prevError);


        //let user adjust learning rate manually to be consistent with other hyper parameters
        // if( fabs(prevError - error) < 0.000001){
        //     std::cout << "Error difference = " << fabs(error - prevError) << " \t\t\t\t--- adjusting learning rates... " <<  std::endl;
        //     trainer.AdjustLearningRates();
        // }

        emit train_nf(error, trainer.epoch_count);

    } while (error > _training_params.error_threshold);
}
