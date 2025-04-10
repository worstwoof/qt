#include "widget.h"
#include "ui_widget.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <QLabel>
#include <QApplication>
#include <QFile>
#include <QMouseEvent>
#include <QMainWindow>
#include <QDebug>
#include <QTimer>
#include <QImage>

using namespace cv;
using namespace std;

Widget::Widget(QWidget *parent)
    : QLabel(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setWindowFlag(Qt::FramelessWindowHint);
    this->installEventFilter(this);

    connect(ui->pushButton_close,&QPushButton::clicked,[=](){
        this->close();
    });

    QFile file("://assets/qtWidget.qss");
    if(file.open(QFile::OpenModeFlag::ReadOnly)){
        this->setStyleSheet(file.readAll());
    }

    timer   = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readfarme()));
    connect(ui->pushButton_open, SIGNAL(clicked()), this, SLOT(opencamara()));
    connect(ui->pushButton_closecamara, SIGNAL(clicked()), this, SLOT(closecamara()));
}

void Widget::opencamara()
{
    capture.open(0);
    qDebug("open");
    if(!capture.isOpened()){
        qDebug("error");
    }
    timer->start(20);
}

void Widget::readfarme()
{
    if(!capture.isOpened()){
        qDebug("error");
    }
    capture>>cap;
    if(!cap.empty()){
        imag=Mat2QImage(cap);
        imag=imag.mirrored(true,false);
        imag=imag.scaled(ui->labelcamera->width(),ui->labelcamera->height(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
        ui->labelcamera->setPixmap(QPixmap::fromImage(imag));
    }
    else
        qDebug("can not");
}


void Widget::closecamara()
{
    timer->stop();
    ui->labelcamera->clear();
}

QImage  Widget::Mat2QImage(Mat cvImg)
{
    QImage qImg;
    if(cvImg.channels()==3)     //3 channels color image
    {

        cv::cvtColor(cvImg,cvImg,cv::COLOR_BGR2RGB);
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols, cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_RGB888);
    }
    else if(cvImg.channels()==1)                    //grayscale image
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols,cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_Indexed8);
    }
    else
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                    cvImg.cols,cvImg.rows,
                    cvImg.cols*cvImg.channels(),
                    QImage::Format_RGB888);
    }
    return qImg;
}


Widget::~Widget()
{
    delete ui;
}



bool Widget::eventFilter(QObject *watched, QEvent *event)
{
    if(watched==this)
    {
        QMouseEvent* MouseEvent = dynamic_cast<QMouseEvent*>(event);
        if(MouseEvent)
        {
            static QPoint offset;
            if(MouseEvent->type()==QEvent::Type::MouseButtonPress)
            {
                offset = MouseEvent->globalPos()-this->pos();
            }
            else if(MouseEvent->type()==QEvent::Type::MouseMove)
            {
                this->move( MouseEvent->globalPos()- offset);
            }
        }
    }




    return QLabel::eventFilter(watched,event);
}




