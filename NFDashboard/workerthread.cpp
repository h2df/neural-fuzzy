#include "workerthread.h"

WorkerThread::WorkerThread(QObject *parent, NFTrainer *trainer, const TrainingDataParams& training_data_params) :QThread(parent),
    trainer(trainer), _data_params(training_data_params)
{
}

void WorkerThread::run(){
    trainer->Initialize(_data_params);
    if (!trainer->HasTrainingDataReady()) {
        emit warning("Invalid training data path: " + _data_params.training_data_path);
        return;
    }

    double training_error = 1.0;
    double validation_error = 1.0;
    double prev_validation_error = 0;

    qDebug() << "============================================" ;
    qDebug() << "Training begins... ";
    do
    {
        if (trainer->ForceStopTraining()) {
            emit beyond_epoch_limit(trainer->epoch_count);
            return;
        }

        trainer->TrainOneEpoch();
        training_error = trainer->CalcTrainingError();
        prev_validation_error = validation_error;
        validation_error = trainer->CalcValidationError();
        qDebug() << "\n\t Training Error: " << training_error;
        qDebug() << "\n\t Validation Error: " << validation_error ;
        qDebug() << "---<< Epoch # " << trainer->epoch_count << " >>---" << ", validation error difference = " << fabs(validation_error - prev_validation_error);


        //let user adjust learning rate manually to be consistent with other hyper parameters
        // if( fabs(prevError - error) < 0.000001){
        //     std::cout << "Error difference = " << fabs(error - prevError) << " \t\t\t\t--- adjusting learning rates... " <<  std::endl;
        //     trainer.AdjustLearningRates();
        // }

        emit train_nf(training_error, validation_error, trainer->epoch_count);

    } while (validation_error > trainer->GetErrorThreshold());

    emit train_success(training_error, validation_error, trainer->epoch_count, trainer->GetNN().GetRulesReport());
}
