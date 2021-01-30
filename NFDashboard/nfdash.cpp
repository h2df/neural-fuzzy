#include "nfdash.h"
#include "ui_nfdash.h"


NFDash::NFDash(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::NFDash)
{
    qRegisterMetaType<std::string>();
    ui->setupUi(this);
    ui->warning_lb->setStyleSheet("color: red");
    ui->plot->addGraph();
    ui->plot->graph(0)->setScatterStyle(QCPScatterStyle::ssPlus);
    ui->plot->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->plot->xAxis->setLabel("Epoch");
    ui->plot->yAxis->setLabel("Error");
    ui->plot->setVisible(false);
}

NFDash::~NFDash()
{
    delete ui;
}

void NFDash::on_training_btn_clicked()
{
    clearPlot();
    ui->plot->setVisible(true);

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
    TrainingDataParams training_data_params = {
        training_data_path,
        shuffle
    };    

    worker = new WorkerThread(this, training_params, training_data_params);
    connect(worker, SIGNAL(train_nf(double, unsigned)), this, SLOT(onTrainNF(double, unsigned)));
    connect(worker, SIGNAL(warning(std::string)), this, SLOT(onWarning(std::string)));
    this->worker->start();
}

void NFDash::onTrainNF(double error, unsigned epoch) {
    this->ui->error_lb->setText("Epoch: " + QString::number(epoch) + ", Error: " + QString::number(error));
    errors.append(error);
    epochs.append((double)epoch);
    plot();
}

void NFDash::onWarning(std::string warning){
    this->ui->warning_lb->setText(QString::fromStdString(warning));
    QTimer::singleShot(2000, [&](){ ui->warning_lb->setText("   ");});
}

void NFDash::plot() {
    ui->plot->graph(0)->setData(epochs, errors);
    ui->plot->yAxis->setRangeUpper(errors[0]);
    ui->plot->xAxis->setRangeUpper(epochs.size());
    ui->plot->replot();
    ui->plot->update();
}

void NFDash::clearPlot() {
    errors.clear();
    epochs.clear();
    ui->plot->graph(0)->setData(epochs, errors);
    ui->plot->replot();
    ui->plot->update();
}

