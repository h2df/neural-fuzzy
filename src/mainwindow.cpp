#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "globalVariables.h"


//--------------------------------------

LetterStructure letters[20001];
LetterStructure testPattern;

bool patternsLoadedFromFile;
int MAX_EPOCHS;
double LEARNING_RATE;
//int GLOBAL_COUNTER;
//int NUMBER_OF_PATTERNS;

///////////////////////////////////////////////////////

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //---------------------------------------
    //initialisation of global variables
    //

    //NUMBER_OF_PATTERNS = 20000;

    LEARNING_RATE=0.2;
//    GLOBAL_COUNTER=0;
    patternsLoadedFromFile = false;
    MAX_EPOCHS = 50;

    bp = new Backpropagation;

    //---------------------------------------
    //initialise widgets

    ui->spinBox_training_Epochs->setValue(MAX_EPOCHS);
    ui->horizScrollBar_LearningRate->setValue(int(LEARNING_RATE*100));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_Read_File_clicked()
{
    qDebug() << "\nReading file...";
    QFile file("complete_data_set.txt");
    file.open(QIODevice::ReadOnly | QIODevice::Text);

    if(!file.exists()){
        patternsLoadedFromFile=false;

        qDebug() << "Data file does not exist!";
        return;
    }

    QTextStream in(&file);



    char t;
    char characterSymbol;
    QString line;

    int counterForLetterA=0;
    int counterForLetterB=0;
    int counterForUnknownLetters=0;
    QString lineOfData;
    QString msg;
    int i=0;
    //while(i < NUMBER_OF_TRAINING_PATTERNS){
    while(i < NUMBER_OF_PATTERNS){

        //e.g. T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
        in >> characterSymbol >> t >> letters[i].f[0] >> t >>  letters[i].f[1] >> t >>  letters[i].f[2] >> t >>  letters[i].f[3] >> t >>  letters[i].f[4] >> t >>  letters[i].f[5] >> t >>  letters[i].f[6] >> t >>  letters[i].f[7] >> t >>  letters[i].f[8] >> t >>  letters[i].f[9] >> t >>  letters[i].f[10] >> t >>  letters[i].f[11] >> t >> letters[i].f[12] >> t >> letters[i].f[13] >> t >> letters[i].f[14] >> t >> letters[i].f[15];

        line = in.readLine();

        if(characterSymbol == 'A'){
            letters[i].symbol = LETTER_A;
            letters[i].outputs[0] = 1;
            letters[i].outputs[1] = 0;
            letters[i].outputs[2] = 0;
            counterForLetterA++;
//            qDebug() << "Letter[" << i << "] = " << characterSymbol;
//            for(int j=0; j < 16; j++){
//                qDebug() << "f[" << j << "] = " << letters[i].f[j];
//            }
        } else if(characterSymbol == 'B'){
            letters[i].symbol = LETTER_B;
            letters[i].outputs[0] = 0;
            letters[i].outputs[1] = 1;
            letters[i].outputs[2] = 0;
            counterForLetterB++;
//            qDebug() << "Letter[" << i << "] = " << characterSymbol;
//            for(int j=0; j < 16; j++){
//                qDebug() << "f[" << j << "] = " << letters[i].f[j];
//            }
        } else {
            letters[i].symbol = UNKNOWN;
            letters[i].outputs[0] = 0;
            letters[i].outputs[1] = 0;
            letters[i].outputs[2] = 1;
            counterForUnknownLetters++;
//            qDebug() << "Unknown Letter[" << i << "] = " << characterSymbol;
//            for(int j=0; j < 16; j++){
//                qDebug() << "f[" << j << "] = " << letters[i].f[j];
//            }
        }

        if((i % 50==0) || (i == NUMBER_OF_PATTERNS-1)){
            msg.clear();
            lineOfData.sprintf("number of patterns for Letter A = %d\n", counterForLetterA);
            msg.append(lineOfData);

            lineOfData.sprintf("number of patterns for Letter B = %d\n", counterForLetterB);
            msg.append(lineOfData);

            lineOfData.sprintf("number of patterns for UNKNOWN letters = %d\n", counterForUnknownLetters);
            msg.append(lineOfData);

            ui->plainTextEdit_results->setPlainText(msg);
            qApp->processEvents();
        }

        i++;
    }

    msg.append("done.");

    ui->plainTextEdit_results->setPlainText(msg);
    qApp->processEvents();

    patternsLoadedFromFile=true;

}

void MainWindow::on_horizScrollBar_LearningRate_valueChanged(int value)
{
    ui->lcdNumber_LearningRate->setSegmentStyle(QLCDNumber::Filled);
    ui->lcdNumber_LearningRate->display(value/1000.0);
    LEARNING_RATE = value/1000.0;
}

//void MainWindow::on_pushButton_Train_Network_clicked()
//{

//}

void MainWindow::on_pushButton_Classify_Test_Pattern_clicked()
{

    char characterSymbol, t;
    QString *q;
    double* classificationResults;

    double* outputs;
    outputs = new double[OUTPUT_NEURONS];

//    delete q;
//    delete classificationResults;



    classificationResults = new double[OUTPUT_NEURONS];

    //QTextStream line;
    q = new QString(ui->plainTextEdit_Input_Pattern->toPlainText());

    QTextStream line(q);

    line >> characterSymbol >> t >> testPattern.f[0] >> t >>  testPattern.f[1] >> t >>  testPattern.f[2] >> t >>  testPattern.f[3] >> t >>  testPattern.f[4] >> t >>  testPattern.f[5] >> t >>  testPattern.f[6] >> t >>  testPattern.f[7] >> t >>  testPattern.f[8] >> t >>  testPattern.f[9] >> t >>  testPattern.f[10] >> t >>  testPattern.f[11] >> t >> testPattern.f[12] >> t >> testPattern.f[13] >> t >> testPattern.f[14] >> t >> testPattern.f[15];



    if(characterSymbol == 'A'){
        testPattern.symbol = LETTER_A;
        testPattern.outputs[0] = 1;
        testPattern.outputs[1] = 0;
        testPattern.outputs[2] = 0;

    } else if(characterSymbol == 'B'){
        testPattern.symbol = LETTER_B;
        testPattern.outputs[0] = 0;
        testPattern.outputs[1] = 1;
        testPattern.outputs[2] = 0;

    } else {
        testPattern.symbol = UNKNOWN;
        testPattern.outputs[0] = 0;
        testPattern.outputs[1] = 0;
        testPattern.outputs[2] = 1;

    }


    //---------------------------------
    classificationResults = bp->testNetwork(testPattern);

    ui->lcdNumber_A->display(classificationResults[0]);
    ui->lcdNumber_B->display(classificationResults[1]);
    ui->lcdNumber_unknown->display(classificationResults[2]);


    //-----------------------------------------------------------
    for(int k=0; k < OUTPUT_NEURONS; k++){
       outputs[k] = testPattern.outputs[k];
    }
    //-----------------------------------------------------------
     QString textClassification;
     switch(bp->action(outputs)){
        case 0:
            textClassification = "letter A";
            break;
        case 1:
            textClassification = "letter B";
            break;
        case 2:
            textClassification = "unknown";
            break;
     };

    if (bp->action(classificationResults) == bp->action(outputs)) {
        qDebug() << "correct classification.";
        ui->label_Classification->setText(textClassification + ", - Correct classification!");
    } else {
        qDebug() << "incorrect classification.";
        ui->label_Classification->setText(textClassification + ", -XXX- Incorrect classification.");
    }

}

void MainWindow::on_pushButton_Train_Network_Max_Epochs_clicked()
{
    double SSE = 0.0;
    QString msg;

    if(!patternsLoadedFromFile) {
        msg.clear();
        msg.append("\nMissing training patterns.  Load data set first.\n");
        ui->plainTextEdit_results->setPlainText(msg);
        return;
    }

    MAX_EPOCHS = ui->spinBox_training_Epochs->value();
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    int e=0;
    for(int i=0; i < MAX_EPOCHS; i++){
      msg.clear();
      msg.append("\nTraining in progress...\n");


//      qApp->processEvents();


      SSE = bp->trainNetwork(); //trains for 1 epoch
      ui->lcdNumber_SSE->display(SSE);

      qApp->processEvents();

      update();
      e++;
      qDebug() << "epoch: " << e << ", SSE = " << SSE;
      msg.append("\nEpoch=");
      msg.append(QString::number(e));
      ui->plainTextEdit_results->setPlainText(msg);

      if((i > 0) && ((i % 5) ==0)) {
         bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

         ui->plainTextEdit_results->setPlainText("Weights saved into file.");
         qApp->processEvents();
      }

    }
    QApplication::restoreOverrideCursor();

}

void MainWindow::on_pushButton_Initialise_Network_clicked()
{
    bp->initialise();
}

void MainWindow::on_pushButton_Test_All_Patterns_clicked()
{
    char characterSymbol;

    double* classificationResults;
    double* outputs;
    int correctClassifications=0;

    classificationResults = new double[OUTPUT_NEURONS];
    outputs = new double[OUTPUT_NEURONS];

    for(int i=NUMBER_OF_TRAINING_PATTERNS; i < NUMBER_OF_PATTERNS; i++){

            characterSymbol = letters[i].symbol;
            for(int j=0; j < INPUT_NEURONS; j++){
                testPattern.f[j] = letters[i].f[j];
            }

            if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_A;
                testPattern.outputs[0] = 1;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 0;

            } else if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_B;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 1;
                testPattern.outputs[2] = 0;

            } else {
                testPattern.symbol = UNKNOWN;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 1;

            }


            //---------------------------------
            classificationResults = bp->testNetwork(testPattern);

            for(int k=0; k < OUTPUT_NEURONS; k++){
               outputs[k] = testPattern.outputs[k];
            }

            if (bp->action(classificationResults) == bp->action(outputs)) {
                 correctClassifications++;

            }


        }


      qDebug() << "TEST SET: correctClassifications = " << correctClassifications;

}

void MainWindow::on_pushButton_Save_Weights_clicked()
{
    bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

    QString msg;
    QString lineOfText;

    lineOfText = "weights saved to file: " + ui->plainTextEdit_saveWeightsAs->toPlainText();

    msg.append(lineOfText);

    ui->plainTextEdit_results->setPlainText(msg);

}

void MainWindow::on_pushButton_Load_Weights_clicked()
{
   bp->loadWeights(ui->plainTextEdit_fileNameLoadWeights->toPlainText());

   QString msg;

   msg.append("weights loaded.\n");

   ui->plainTextEdit_results->setPlainText(msg);


}

void MainWindow::on_pushButton_testNetOnTrainingSet_clicked()
{
    char characterSymbol;
//    QString *q;
    double* classificationResults;
    double* outputs;
    int correctClassifications=0;

    classificationResults = new double[OUTPUT_NEURONS];
    outputs = new double[OUTPUT_NEURONS];

    for(int i=0; i < NUMBER_OF_TRAINING_PATTERNS; i++){

            characterSymbol = letters[i].symbol;
            for(int j=0; j < INPUT_NEURONS; j++){
                testPattern.f[j] = letters[i].f[j];
            }

            if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_A;
                testPattern.outputs[0] = 1;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 0;

            } else if(characterSymbol == LETTER_B){
                testPattern.symbol = LETTER_B;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 1;
                testPattern.outputs[2] = 0;

            } else {
                testPattern.symbol = UNKNOWN;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 1;

            }

            //---------------------------------
            classificationResults = bp->testNetwork(testPattern);

            for(int k=0; k < OUTPUT_NEURONS; k++){
               outputs[k] = testPattern.outputs[k];
            }

            if (bp->action(classificationResults) == bp->action(outputs)) {
                 correctClassifications++;
            }

        }
      qDebug() << "TRAINING SET: correctClassifications = " << correctClassifications;
}

void MainWindow::on_horizScrollBar_LearningRate_actionTriggered(int action)
{

}
