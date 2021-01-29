#include "nfdash.h"
#include "ui_nfdash.h"


NFDash::NFDash(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::NFDash)
{
    qRegisterMetaType<std::string>();
    ui->setupUi(this);
    ui->warning_lb->setStyleSheet("color: red");
}

NFDash::~NFDash()
{
    delete ui;
}

void NFDash::on_training_btn_clicked()
{
    double weight_learning_rate, func_center_learning_rate, func_width_learning_rate;
    unsigned center_move_iterate;
    double initial_rule_weight;
    unsigned rule_num;
    double error_threshold;
    unsigned max_epoch;
    bool use_validation;

    weight_learning_rate = this->ui->rule_weight_lr_text->toPlainText().toDouble();
    func_center_learning_rate = this->ui->center_lr_text->toPlainText().toDouble();
    func_width_learning_rate = this->ui->width_lr_text->toPlainText().toDouble();
    center_move_iterate = this->ui->center_iterate_text->toPlainText().toUInt();
    initial_rule_weight = this->ui->ini_rule_weight_combo->currentText().toDouble();
    rule_num = this->ui->rule_num_combo->currentText().toUInt();
    error_threshold = this->ui->threshold_txt->toPlainText().toDouble();
    max_epoch = this->ui->max_epoch_txt->toPlainText().toUInt();
    use_validation = this->ui->validation_checkbox->isChecked();

    NFTrainParams training_params = {
        weight_learning_rate, func_center_learning_rate, func_width_learning_rate,
        center_move_iterate,
        initial_rule_weight,
        rule_num,
        error_threshold,
        max_epoch,
        use_validation
    };

    std::string training_data_path = this->ui->training_data_text->toPlainText().toStdString();
    bool shuffle = this->ui->shuffle_checkbox->isChecked();
    bool normalize = this->ui->normalize_checkbox->isChecked();
    TrainingDataParams training_data_params = {
        training_data_path,
        shuffle, normalize
    };




    worker = new WorkerThread(this, training_params, training_data_params);
    connect(worker, SIGNAL(train_nf(double)), this, SLOT(onTrainNF(double)));
    connect(worker, SIGNAL(warning(std::string)), this, SLOT(onWarning(std::string)));
    this->worker->start();
}

void NFDash::onTrainNF(double error) {
    this->ui->error_txt->setText(QString::number(error));
}

void NFDash::onWarning(std::string warning){
    this->ui->warning_lb->setText(QString::fromStdString(warning));
    QTimer::singleShot(2000, [&](){ ui->warning_lb->setText("   ");});
}

