#ifndef NFDASH_H
#define NFDASH_H

#include <QMainWindow>
#include "trainthread.h"
#include "testthread.h"
#include "algorithm.h"
#include <QtDebug>
#include <QFileDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class NFDash; }
QT_END_NAMESPACE
Q_DECLARE_METATYPE(std::string)
class NFDash : public QMainWindow
{
    Q_OBJECT

public:
    NFDash(QWidget *parent = nullptr);
    ~NFDash();
    TrainThread *train_thread;
    TestThread *test_thread;



private slots:
    void on_training_btn_clicked();
    void onTrainNF(double, double);
    void onWarning(std::string);
    void onBeyondEpochLimit(unsigned);
    void on_ld_training_data_btn_clicked();
    void onTrainSuccess(double, double);

    void on_shuffle_checkbox_stateChanged(int arg1);

    void on_ld_test_data_btn_clicked();

    void on_test_btn_clicked();

private:
    Ui::NFDash *ui;
    QVector<double> validation_errors;
    QVector<double> training_errors;
    QVector<double> epochs;
    NFTrainer trainer;
    PendulumDataNormalizer normalizer;

    void plot();
    void clearPlot();
    void scaleBackPlot();
    void clearRuleReport();
};
#endif // NFDASH_H
