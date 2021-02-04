#include "trainthread.h"


TrainThread::TrainThread(QObject *parent, NFTrainer *trainer, PendulumDataNormalizer *normalizer, const std::string training_data_path) :QThread(parent),
    trainer(trainer), normalizer(normalizer), training_data_path(training_data_path)
{
}

void TrainThread::run(){

    if (!Initialize()){
        emit warning("Invalid training data path: " + training_data_path);
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

        emit train_nf(training_error, validation_error);

    } while (validation_error > trainer->GetErrorThreshold());

    emit train_success(training_error, validation_error);
}


bool TrainThread::Initialize()
{

    if (!trainer->Initialize(training_data_path)) {
        return false;
    }

    normalizer->Initialize(trainer->GetTrainingData());
    trainer->NormalizeData(normalizer);
    return true;
}
