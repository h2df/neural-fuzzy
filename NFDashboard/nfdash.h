#ifndef NFDASH_H
#define NFDASH_H

#include <QMainWindow>
#include "workerthread.h"
#include "algorithm.h"
#include <QtDebug>

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
    WorkerThread *worker;
    NFSystem nn;


private slots:
    void on_training_btn_clicked();
    void onTrainNF(double);
    void onWarning(std::string);
private:
    Ui::NFDash *ui;
};
#endif // NFDASH_H
