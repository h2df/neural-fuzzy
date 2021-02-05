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
    delete train_thread;
    delete test_thread;
}

void NFDash::on_training_btn_clicked()
{
    clearPlot();
    clearRuleReport();
    ui->plot->setVisible(true);

    double weight_learning_rate, func_center_learning_rate, func_width_learning_rate;
    unsigned center_move_iterate;
    double initial_rule_weight;
    unsigned rule_num;
    double error_threshold;
    unsigned max_epoch;
    double validation_factor;
    bool shuffle;

    weight_learning_rate = this->ui->rule_weight_lr_spin->value();
    func_center_learning_rate = this->ui->func_center_lr_spin->value();
    func_width_learning_rate = this->ui->func_width_lr_spin->value();
    center_move_iterate = this->ui->func_center_iterate_spin->value();
    initial_rule_weight = this->ui->ini_rule_weight_combo->currentText().toDouble();
    rule_num = this->ui->rule_num_combo->currentText().toUInt();
    error_threshold = this->ui->error_threshold_spin->value();
    max_epoch = this->ui->max_epoch_spin->value();
    validation_factor = this->ui->validation_factor_spin->value();
    shuffle = this->ui->shuffle_checkbox->isChecked();
    if (shuffle) {
        srand(ui->seed_spin->value());
    }

    nf = new NFSystem(rule_num, initial_rule_weight);

    NFTrainParams training_params = {
        weight_learning_rate, func_center_learning_rate, func_width_learning_rate,
        center_move_iterate,
        error_threshold,
        max_epoch,
        validation_factor,
        shuffle
    };

    std::string training_data_path = ui->training_data_path_lb->text().toStdString();

    train_thread = new TrainThread(this, nf, training_params, normalizer, training_data_path);
    connect(train_thread, SIGNAL(train_nf(double, double, unsigned)), this, SLOT(onTrainNF(double, double, unsigned)));
    connect(train_thread, SIGNAL(warning(std::string)), this, SLOT(onWarning(std::string)));
    connect(train_thread, SIGNAL(beyond_epoch_limit(unsigned)), this, SLOT(onBeyondEpochLimit(unsigned)));
    connect(train_thread, SIGNAL(train_success(double, double, unsigned, std::string)), this,  SLOT(onTrainSuccess(double, double, unsigned, std::string)));
    this->train_thread->start();
}

void NFDash::onTrainNF(double training_error, double validation_error, unsigned epoch_count) {
    this->ui->error_lb->setText("Epoch: " + QString::number(epoch_count) + ", Training Error: " + QString::number(training_error) + ", Validation Error: " + QString::number(validation_error));
    training_errors.append(training_error);
    validation_errors.append(validation_error);
    epochs.append(double (epoch_count));
    plot();
}

void NFDash::onTestNF(double error)
{
   ui->error_lb->setText("Test Error: " + QString::number(error));
}

void NFDash::onWarning(std::string warning){
    this->ui->warning_lb->setText(QString::fromStdString(warning));
    QTimer::singleShot(2000, [&](){ ui->warning_lb->setText("   ");});
}

void NFDash::onBeyondEpochLimit(unsigned epoch)
{
    onWarning(std::to_string(epoch) + " epochs is beyond the limit " + std::to_string(ui->max_epoch_spin->value()) + ". Training is unsuccessful.");
}

void NFDash::plot() {
    ui->plot->graph(0)->setData(epochs, training_errors);
    ui->plot->graph(1)->setData(epochs, validation_errors);
    unsigned idx = validation_errors.size() / 50 * 49;
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

void NFDash::clearRuleReport() {
    ui->rules_text->clear();
}


void NFDash::on_ld_training_data_btn_clicked()
{
    QString fn = QFileDialog::getOpenFileName(this, "Open training data file", "C:\\Users\\zhang\\Desktop\\159333Data");
    ui->training_data_path_lb->setText(fn);
    ui->training_btn->setEnabled(true);
}

void NFDash::on_ld_test_data_btn_clicked()
{
    QString fn = QFileDialog::getOpenFileName(this, "Open test data file", "C:\\Users\\zhang\\Desktop\\159333Data");
    ui->test_data_path_lb->setText(fn);
    ui->test_btn->setEnabled(true);
}

void NFDash::scaleBackPlot()
{
    ui->plot->yAxis->setRangeUpper(validation_errors[0]);
    ui->plot->replot();
    ui->plot->update();
}

void NFDash::onTrainSuccess(double training_error, double validation_error, unsigned epoch_count, std::string report)
{
    scaleBackPlot();
    ui->error_lb->setText("Successfully trained after " + QString::number(epoch_count) + " epochs. The average error on training data is " + QString::number(training_error) + " and the average error on validation data is " + QString::number(validation_error));
    ui->rules_text->setText(QString::fromStdString(report));
}

void NFDash::on_shuffle_checkbox_stateChanged(int checked)
{
    ui->seed_lb->setVisible(checked);
    ui->seed_spin->setVisible(checked);
}


void NFDash::on_test_btn_clicked()
{
    std::string tet_data_path = ui->test_data_path_lb->text().toStdString();
    test_thread = new TestThread(this, nf, tet_data_path);
    connect(test_thread, SIGNAL(warning(std::string)), this, SLOT(onWarning(std::string)));
    connect(test_thread, SIGNAL(test_nf(double)), this, SLOT(onTestNF(double)));
    test_thread->start();
}
