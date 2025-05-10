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
    : QLabel(parent),
    ui(new Ui::Widget)
    , confThreshold(0.4f)

                                       // 0.01f 到 0.05f 可能是比较柔和的开始。
    , nmsThreshold(0.4f)

    , inpWidth(416)
    , inpHeight(416)

    , frameCounter(0)       // 初始化帧计数器为 0
       , processInterval(13)    //  初始化处理间隔为 3 (表示每3帧处理一次YOLO，你可以改成2, 4, 5等试试效果)
 , particle_manager_(processInterval)

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
        qDebug() << "Trying to load YOLO model...";
            qDebug() << "Config file:" << QString::fromStdString(modelConfiguration);
            qDebug() << "Weights file:" << QString::fromStdString(modelWeights);
            qDebug() << "Classes file:" << QString::fromStdString(classesFile);
    QFile file("://assets/qtWidget.qss");
    if(file.open(QFile::OpenModeFlag::ReadOnly)){
        this->setStyleSheet(file.readAll());
    }
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
      if (net.empty()) {
          qDebug() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
          qDebug() << "CRITICAL ERROR: YOLO Network is EMPTY after loading!";
          qDebug() << "Please check if the .cfg and .weights files are correct and in the expected location (usually the build/debug or build/release folder).";
          qDebug() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
          // 在这里可以考虑直接 return 或者设置一个错误状态，阻止程序继续
      } else {
          qDebug() << "DEBUG: YOLO Network loaded successfully (not empty).";

      }

      qDebug() << "------------------------------------------------------";
      qDebug() << "DEBUG: Attempting to load classes file...";
      qDebug() << "DEBUG: Classes file path:" << QString::fromStdString(classesFile);

      std::ifstream ifs(classesFile.c_str());
      if (!ifs.is_open()) {
          qDebug() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
          qDebug() << "CRITICAL ERROR: Cannot open classes file:" << QString::fromStdString(classesFile);
          qDebug() << "Please check if the .names file is correct and in the expected location.";
          qDebug() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
      } else {
          classNames.clear(); // 清空旧的类别名
          std::string line;
          while (std::getline(ifs, line)) {
              classNames.push_back(line);
          }
          ifs.close();
          qDebug() << "DEBUG: Classes loaded successfully.";
          qDebug() << "DEBUG: Number of classes found:" << classNames.size();
          if (!classNames.empty()) {
              for (int i = 0; i < classNames.size(); ++i) {
                  qDebug() << "DEBUG: Class ID" << i << ":" << QString::fromStdString(classNames[i]);
              }
          } else {
              qDebug() << "WARNING: classNames list is empty after loading!";
          }
      }
      qDebug() << "======================================================";
    timer   = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readfarme()));
    connect(ui->pushButton_open, SIGNAL(clicked()), this, SLOT(opencamara()));
    connect(ui->pushButton_closecamara, SIGNAL(clicked()), this, SLOT(closecamara()));
    connect(ui->pushButton_lizi, &QPushButton::clicked, this, &Widget::on_pushButton_lizi_clicked);
}



void Widget::opencamara()
{
    capture.open(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    qDebug("open");
    if(!capture.isOpened()){
        qDebug("error");
    }
    timer->start(50);
}
void Widget::readfarme()
{
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    if (net.empty()) {
        qDebug() << "ERROR";
        cv::Mat errorFrame = cv::Mat::zeros(cv::Size(ui->labelcamera->width(), ui->labelcamera->height()), CV_8UC3);
        cv::putText(errorFrame, "Error", cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
        imag = Mat2QImage(errorFrame);
        ui->labelcamera->setPixmap(QPixmap::fromImage(imag));
        return;
    }

    cv::Mat frame;
    capture >> frame;
    auto time_after_capture = std::chrono::high_resolution_clock::now();
    if (frame.empty()) {
        qDebug() << "没打开摄像头";
        return;
    }
    frameCounter++;


       if (frameCounter % processInterval == 0) // 如果当前帧数是处理间隔的倍数
       {

           if (net.empty()) { // 再次检查模型是否加载
               qDebug() << "YOLO模型还没准备好，readfarme() 本轮YOLO处理跳过...";
               // 这里可以选择是否在跳过YOLO处理时清空上一帧的检测结果，
               // 或者保持上一帧的检测结果显示。目前我们只在处理时画框。
           } else {

               cv::Mat blob;
               cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
               net.setInput(blob);
               std::vector<cv::Mat> outs;
               net.forward(outs, getOutputsNames(net)); // 执行前向传播
               postprocess(frame, outs);


           }

       }
       auto time_after_yolo_postprocess = std::chrono::high_resolution_clock::now();
       particle_manager_.updateAndDrawParticles(frame);

       auto time_after_particles = std::chrono::high_resolution_clock::now();
    // 将处理后的图像显示到界面上
    imag = Mat2QImage(frame);
    imag=imag.mirrored(true,false);
    auto time_after_display = std::chrono::high_resolution_clock::now();
    auto capture_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_capture - overall_start_time).count();
    auto yolo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_yolo_postprocess - time_after_capture).count();
    auto particle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_particles - time_after_yolo_postprocess).count();
    auto display_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_display - time_after_particles).count();
    auto total_frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_display - overall_start_time).count();

    qDebug() << "Frame Times (ms): Total=" << total_frame_duration
             << "Capture=" << capture_duration
             << "YOLO=" << yolo_duration
             << "Particles=" << particle_duration
             << "Display=" << display_duration;
    if (ui->labelcamera) {
         ui->labelcamera->setPixmap(QPixmap::fromImage(imag).scaled(ui->labelcamera->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else if (this->inherits("QLabel")) { // 如果 Widget 本身就是 QLabel
         this->setPixmap(QPixmap::fromImage(imag).scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    this->update();
}


void Widget::closecamara()
{
    timer->stop();
    if (capture.isOpened()) {
        capture.release();
        qDebug() << "摄像头已关闭，下次再见！";
    }
    // 清空一下显示的图像
    if (ui->labelcamera) {
        QPixmap blankPixmap(ui->labelcamera->size()) ;
        blankPixmap.fill(Qt::black);
        ui->labelcamera->setPixmap(blankPixmap);
    } else if (this->inherits("QLabel")) {
        QPixmap blankPixmap(this->size());
        blankPixmap.fill(Qt::black);
        this->setPixmap(blankPixmap);
    }
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
    if (capture.isOpened()) {
            capture.release();
        }
}
std::vector<std::string> Widget::getOutputsNames(const cv::dnn::Net& net_param)
{
    static std::vector<std::string> names; // static 缓存结果，不用每次都算
    if (names.empty())
    {
        // 获取所有层的名字
        std::vector<std::string> layersNames = net_param.getLayerNames();
        // 获取输出层的索引
        std::vector<int> outLayers = net_param.getUnconnectedOutLayers();
        // 根据索引找到输出层的名字
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
        // outs[i] 是一个 Mat，每一行是一个检测到的物体
        // 数据格式: [center_x, center_y, width, height, object_confidence, class1_score, class2_score, ...]
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols); // 从第5个元素开始是类别分数
            cv::Point classIdPoint;
            double confidence;
            // 找到分数最高的类别和它的置信度
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) // 如果置信度大于我们设的阈值
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x); // 记录类别ID
                confidences.push_back((float)confidence); // 记录置信度
                boxes.push_back(cv::Rect(left, top, width, height)); // 记录框框
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
qDebug() << "Detections this frame:" << indices.size(); // 确认检测数量
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::string label = "Unknown";
        if (!classNames.empty() && classIds[idx] < (int)classNames.size()) {
            label = classNames[classIds[idx]];
        }


               //  计算检测框的中心点
               int centerX_box = box.x + box.width / 2;
               int centerY_box = box.y + box.height / 2;
               cv::Point center_of_box(centerX_box, centerY_box); // 创建一个 Point 对象

               //  画一个小红圆点作为中心点标记
               int radius = 3; // 圆点的半径，可以调整大小
               cv::Scalar red_color(0, 0, 255); // BGR 顺序，所以红色是 (0, 0, 255)
               int thickness = -1;
               cv::circle(frame, center_of_box, radius, red_color, thickness);

               qDebug() << "Center point at: (" << centerX_box << "," << centerY_box << ")";

                if (particle_manager_.isEnabled() && (frameCounter % processInterval == 0)) {
                       qDebug() << "Condition met, calling generateParticles for center:" << centerX_box << centerY_box;
                       particle_manager_.generateParticles(center_of_box); //  为当前检测到的手生成粒子
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
    bool newState = !particle_manager_.isEnabled(); // 获取新状态
        particle_manager_.setEnabled(newState);      // 设置新状态

        if (newState) {
            ui->pushButton_lizi->setText("关闭粒子");
        } else {
            ui->pushButton_lizi->setText("开启粒子");
        }
}
