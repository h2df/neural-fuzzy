#include "nfdash.h"
#include "ui_nfdash.h"


NFDash::NFDash(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::NFDash)
{
    qRegisterMetaType<std::string>();
    ui->setupUi(this);
    ui->warning_lb->setStyleSheet("color: red");
    ui->plot->addGraph();//for training error
    ui->plot->addGraph();//for validation error
    ui->plot->graph(0)->setScatterStyle(QCPScatterStyle::ssPlus);
    ui->plot->graph(0)->setName("Training Error");
    ui->plot->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->plot->graph(1)->setScatterStyle(QCPScatterStyle::ssPlus);
    ui->plot->graph(1)->setLineStyle(QCPGraph::lsLine);
    ui->plot->graph(1)->setName("Validation Error");
    ui->plot->legend->setVisible(true);

    QPen pen;
    pen.setColor(QColor(255, 0, 0));
    ui->plot->graph(1)->setPen(pen);

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
    double validation_factor;

    weight_learning_rate = this->ui->rule_weight_lr_text->toPlainText().toDouble();
    func_center_learning_rate = this->ui->center_lr_text->toPlainText().toDouble();
    func_width_learning_rate = this->ui->width_lr_text->toPlainText().toDouble();
    center_move_iterate = this->ui->center_iterate_text->toPlainText().toUInt();
    initial_rule_weight = this->ui->ini_rule_weight_combo->currentText().toDouble();
    rule_num = this->ui->rule_num_combo->currentText().toUInt();
    error_threshold = this->ui->threshold_txt->toPlainText().toDouble();
    max_epoch = this->ui->max_epoch_txt->toPlainText().toUInt();
    validation_factor = this->ui->validation_factor_spin->value();

    NFTrainParams training_params = {
        weight_learning_rate, func_center_learning_rate, func_width_learning_rate,
        center_move_iterate,
        initial_rule_weight,
        rule_num,
        error_threshold,
        max_epoch,
        validation_factor
    };

    std::string training_data_path = this->ui->training_data_text->toPlainText().toStdString();
    bool shuffle = this->ui->shuffle_checkbox->isChecked();
    TrainingDataParams training_data_params = {
        training_data_path,
        shuffle
    };    

    worker = new WorkerThread(this, training_params, training_data_params);
    connect(worker, SIGNAL(train_nf(double, double, unsigned)), this, SLOT(onTrainNF(double, double, unsigned)));
    connect(worker, SIGNAL(warning(std::string)), this, SLOT(onWarning(std::string)));
    connect(worker, SIGNAL(beyond_epoch_limit(unsigned)), this, SLOT(onBeyondEpochLimit(unsigned)));
    this->worker->start();
}

void NFDash::onTrainNF(double training_error, double validation_error, unsigned epoch) {
    this->ui->error_lb->setText("Epoch: " + QString::number(epoch) + ", Training Error: " + QString::number(training_error) + ", Validation Error: " + QString::number(validation_error));
    training_errors.append(training_error);
    validation_errors.append(validation_error);
    epochs.append(double (epoch));
    plot();
}

void NFDash::onWarning(std::string warning){
    this->ui->warning_lb->setText(QString::fromStdString(warning));
    QTimer::singleShot(2000, [&](){ ui->warning_lb->setText("   ");});
}

void NFDash::onBeyondEpochLimit(unsigned epoch)
{
    onWarning(std::to_string(epoch) + " epochs is beyond the limit " + ui->max_epoch_txt->toPlainText().toStdString() + ". Training is unsuccessful.");
}

void NFDash::plot() {
    ui->plot->graph(0)->setData(epochs, training_errors);
    ui->plot->graph(1)->setData(epochs, validation_errors);
    unsigned idx = validation_errors.size() / 30 * 29;
    ui->plot->yAxis->setRangeUpper(validation_errors[idx]);
    ui->plot->xAxis->setRangeUpper(epochs.size());
    ui->plot->replot();
    ui->plot->update();
}

void NFDash::clearPlot() {
    training_errors.clear();
    validation_errors.clear();
    epochs.clear();
    ui->plot->graph(0)->setData(epochs, training_errors);
    ui->plot->graph(1)->setData(epochs, validation_errors);
    ui->plot->replot();
    ui->plot->update();
}

