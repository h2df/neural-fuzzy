#include "trainthread.h"


TrainThread::TrainThread(QObject *parent, NFSystem *nf, const NFTrainParams params, PendulumDataNormalizer *normalizer, const std::string training_data_path) :QThread(parent),
    nf(nf), params(params), normalizer(normalizer), training_data_path(training_data_path)
{
}

void TrainThread::run(){
    NFTrainer trainer(nf, params);
    NFTester tester(nf);

    if (!trainer.Initialize(training_data_path)) {
        emit warning("Invalid training data path: " + training_data_path);
        return;
    }
    normalizer->Initialize(trainer.GetTrainingData());
    trainer.NormalizeData(normalizer);

    double training_error = 1.0;
    double validation_error = 1.0;
    double prev_validation_error = 0;

    qDebug() << "============================================" ;
    qDebug() << "Training begins... ";
    do
    {
        if (trainer.ForceStopTraining()) {
            emit beyond_epoch_limit(trainer.epoch_count);
            return;
        }

        trainer.TrainOneEpoch();
        training_error = tester.CalcAvgError(trainer.GetTrainingData());
        prev_validation_error = validation_error;
        validation_error = tester.CalcAvgError(trainer.GetValidationData());
        qDebug() << "\n\t Training Error: " << training_error;
        qDebug() << "\n\t Validation Error: " << validation_error ;
        qDebug() << "---<< Epoch # " << trainer.epoch_count << " >>---" << ", validation error difference = " << fabs(validation_error - prev_validation_error);

        emit train_nf(training_error, validation_error, trainer.epoch_count);

    } while (validation_error > trainer.GetErrorThreshold());

    emit train_success(training_error, validation_error, trainer.epoch_count, nf->GetRulesReport());
}
