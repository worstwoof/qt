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
#include <QMessageBox>

using namespace cv;
using namespace std;

Widget::Widget(QWidget *parent)
    : QLabel(parent),
      ui(new Ui::Widget),
      confThreshold(0.4f),
      nmsThreshold(0.4f),
      inpWidth(416),
      inpHeight(416),
      frameCounter(0),
      processInterval(13),
      particle_manager_(processInterval)
{
    ui->setupUi(this);
    this->setWindowFlag(Qt::FramelessWindowHint);
    this->installEventFilter(this);

    connect(ui->pushButton_close,&QPushButton::clicked,[=](){
        this->close();
    });

    std::string modelConfiguration = "E:/qtcode/build-qt-Desktop_Qt_5_14_2_MinGW_64_bit-Debug/debug/cross-hands.cfg";
    std::string modelWeights = "E:/qtcode/build-qt-Desktop_Qt_5_14_2_MinGW_64_bit-Debug/debug/cross-hands.weights";
    std::string classesFile = "E:/qtcode/build-qt-Desktop_Qt_5_14_2_MinGW_64_bit-Debug/debug/cross-hands.names";
    QString::fromStdString(modelConfiguration);
    QString::fromStdString(modelWeights);
    QString::fromStdString(classesFile);
    QFile file("://assets/qtWidget.qss");
    if(file.open(QFile::OpenModeFlag::ReadOnly)){
        this->setStyleSheet(file.readAll());
    }
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open()) {
        QString::fromStdString(classesFile);
    } else {
        classNames.clear(); // 清空旧的类别名
        std::string line;
        while (std::getline(ifs, line)) {
            classNames.push_back(line);
        }
        ifs.close();
        if (!classNames.empty()) {
            for (int i = 0; i < classNames.size(); ++i) {
                qDebug() << "DEBUG: Class ID" << i << ":" << QString::fromStdString(classNames[i]);
            }
        } else {
            qDebug() << "WARNING: classNames list is empty after loading!";
        }
    }
    timer   = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readfarme()));
    connect(ui->pushButton_camara, &QPushButton::clicked, this, &Widget::on_pushButton_camera_clicked);
    connect(ui->pushButton_lizi, &QPushButton::clicked, this, &Widget::on_pushButton_lizi_clicked);
    connect(ui->pushButton_snapshot, &QPushButton::clicked, this, &Widget::on_pushButton_snapshot_clicked);
    connect(ui->pushButton_record, &QPushButton::clicked, this, &Widget::on_pushButton_record_clicked);
    ui->pushButton_snapshot->setEnabled(false);
    ui->pushButton_record->setEnabled(false);
    ui->pushButton_lizi->setEnabled(false);
}

void Widget::on_pushButton_snapshot_clicked() {
    const QPixmap* pixmap = ui->labelcamera->pixmap();
    if (pixmap && !pixmap->isNull()) {
        QImage image_to_save = pixmap->toImage();
        QString default_path = QDir::homePath() + "/snapshot_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png";
        QString file_path = QFileDialog::getSaveFileName(this, "保存快照", default_path, "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)");
        if (!file_path.isEmpty()) {
            if (image_to_save.save(file_path)) {
                qDebug() << "Snapshot (from QLabel) saved to:" << file_path;
                QMessageBox::information(this, "成功", "快照已保存到:\n" + file_path);
            } else {
                QMessageBox::warning(this, "错误", "保存快照失败！");
            }
        }
    } else {
        QMessageBox::information(this, "提示", "当前没有图像可以拍照。");
    }
}
void Widget::on_pushButton_record_clicked() {
    if (!is_recording_) {
        if (!capture.isOpened()) {
            QMessageBox::warning(this, "错误", "请先打开摄像头！");
            return;
        }
        QString default_path = QDir::homePath() + "/recording_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".avi";
        video_output_path_ = QFileDialog::getSaveFileName(this, "选择录像保存位置", default_path, "AVI 视频 (*.avi);;MP4 视频 (*.mp4)");
        if (video_output_path_.isEmpty()) {
            qDebug() << "Video recording save cancelled by user.";
            return;
        }
        int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        video_writer_.open(video_output_path_.toStdString(), fourcc, fps, cv::Size(frame_width, frame_height), true);
        is_recording_ = true;
        ui->pushButton_record->setText("停止录像");
    } else {
        is_recording_ = false;
        if (video_writer_.isOpened()) {
            video_writer_.release();
            QMessageBox::information(this, "成功", "录像已保存到:\n" + video_output_path_);
        }
        ui->pushButton_record->setText("开始录像");
    }
}
void Widget::on_pushButton_camera_clicked() {
    if (!camera_is_opened_) {
        qDebug() << "Attempting to open camera...";
        if (capture.open(0)) {
            camera_is_opened_ = true;
            timer->start(30);
            ui->pushButton_camara->setText("关闭摄像头");
            qDebug() << "Camera opened successfully.";
            ui->pushButton_snapshot->setEnabled(true);
            ui->pushButton_record->setEnabled(true);
            ui->pushButton_lizi->setEnabled(true);

        } else {
            qDebug() << "ERROR: Failed to open camera!";
            QMessageBox::critical(this, "错误", "无法打开摄像头！");
        }
    } else {
        timer->stop();
        if (is_recording_ && video_writer_.isOpened()) {
            is_recording_ = false;
            video_writer_.release();
            qDebug() << "Recording stopped and saved due to camera close.";
            ui->pushButton_record->setText("开始录像");
            QMessageBox::information(this, "提示", "摄像头已关闭，录像已自动保存。");
        }

        if (capture.isOpened()) {
            capture.release();
        }
        camera_is_opened_ = false;
        ui->pushButton_camara->setText("打开摄像头");
        qDebug() << "Camera closed.";

        if (ui->labelcamera) {
            QPixmap blankPixmap(ui->labelcamera->size());
            blankPixmap.fill(Qt::black);
            ui->labelcamera->setPixmap(blankPixmap);
        }
        ui->pushButton_snapshot->setEnabled(false);
        ui->pushButton_record->setEnabled(false);
        ui->pushButton_lizi->setEnabled(false);
        if(particle_manager_.isEnabled()){
            particle_manager_.setEnabled(false);
            ui->pushButton_lizi->setText("开启粒子");
        }
    }
}

void Widget::readfarme()
{
    cv::Mat frame;
    capture >> frame;
    if (frame.empty()) {
        qDebug() << "Frame is empty!";
        if (is_recording_ && video_writer_.isOpened()) {
            on_pushButton_record_clicked();
        }
        return;
    }
    frameCounter++;
    if (frameCounter % processInterval == 0)
    {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames(net));
        postprocess(frame, outs);
    }
    particle_manager_.updateAndDrawParticles(frame);
    imag = Mat2QImage(frame);
    imag=imag.mirrored(true,false);
    if (is_recording_ && video_writer_.isOpened()) {
        video_writer_.write(frame);
    }
    cv::flip(frame, frame, 1);
    if (ui->labelcamera) {
        ui->labelcamera->setPixmap(QPixmap::fromImage(imag).scaled(ui->labelcamera->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else if (this->inherits("QLabel")) {
        this->setPixmap(QPixmap::fromImage(imag).scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    this->update();
}
QImage  Widget::Mat2QImage(Mat cvImg)
{
    QImage qImg;
    if(cvImg.channels()==3)
    {

        cv::cvtColor(cvImg,cvImg,cv::COLOR_BGR2RGB);
        qImg =QImage((const unsigned char*)(cvImg.data),
                     cvImg.cols, cvImg.rows,
                     cvImg.cols*cvImg.channels(),
                     QImage::Format_RGB888);
    }
    else if(cvImg.channels()==1)
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
    if (is_recording_ && video_writer_.isOpened()) { // 如果退出时还在录像
        video_writer_.release();
        qDebug() << "Recording stopped and saved due to widget destruction.";
    }
    delete ui;
}
std::vector<std::string> Widget::getOutputsNames(const cv::dnn::Net& net_param)
{
    static std::vector<std::string> names; // static 缓存结果，不用每次都算
    if (names.empty())
    {
        std::vector<std::string> layersNames = net_param.getLayerNames();
        std::vector<int> outLayers = net_param.getUnconnectedOutLayers();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1]; //层索引是从1开始的
    }
    return names;
}
void Widget::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // 从第5个元素开始是类别分数
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    qDebug() << "Detections this frame:" << indices.size();
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::string label = "Unknown";
        if (!classNames.empty() && classIds[idx] < (int)classNames.size()) {
            label = classNames[classIds[idx]];
        }
        int centerX_box = box.x + box.width / 2;
        int centerY_box = box.y + box.height / 2;
        cv::Point center_of_box(centerX_box, centerY_box);
        int radius = 3;
        cv::Scalar red_color(0, 0, 255);
        int thickness = -1;
        cv::circle(frame, center_of_box, radius, red_color, thickness);
        if (particle_manager_.isEnabled() && (frameCounter % processInterval == 0)) {
            particle_manager_.generateParticles(center_of_box);
        }
    }
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
void Widget::on_pushButton_lizi_clicked()
{
    bool newState = !particle_manager_.isEnabled();
    particle_manager_.setEnabled(newState);
    if (newState) {
        ui->pushButton_lizi->setText("关闭粒子");
    } else {
        ui->pushButton_lizi->setText("开启粒子");
    }
}
