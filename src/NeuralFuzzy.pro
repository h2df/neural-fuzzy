#-------------------------------------------------
#
# Project created by QtCreator 2013-09-18T15:17:01
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NeuralFuzzy
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    backpropagation.cpp

HEADERS  += mainwindow.h \
    globalVariables.h \
    backpropagation.h

FORMS    += mainwindow.ui
