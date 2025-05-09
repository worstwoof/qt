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
    , confThreshold(0.5f)

                                       // 0.01f 到 0.05f 可能是比较柔和的开始。
    , nmsThreshold(0.4f)

    , inpWidth(416)
    , inpHeight(416)

    , frameCounter(0)       // 初始化帧计数器为 0
       , processInterval(8)    //  初始化处理间隔为 3 (表示每3帧处理一次YOLO，你可以改成2, 4, 5等试试效果)
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
    std::ifstream ifs(classesFile.c_str());
        if (!ifs.is_open()) {
            qDebug() << " ERROR: Cannot open classes file:" << QString::fromStdString(classesFile);
        } else {
            std::string line;
            while (std::getline(ifs, line)) {
                classNames.push_back(line);
            }
            qDebug() << "Classes loaded! Number of classes:" << classNames.size();
            for(const auto& name : classNames) {
                qDebug() << "   - " << QString::fromStdString(name);
            }
        }
        ifs.close();


        net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        if (net.empty()) {
            qDebug() << "CRITICAL ERROR: Cannot load YOLO network!";
            qDebug() << "   Make sure '" << QString::fromStdString(modelConfiguration) << "' and '" << QString::fromStdString(modelWeights) << "' are in the build directory!";
        } else {
            qDebug() << "Hooray! YOLO Network loaded successfully!";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
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
    timer->start(60);
}
void Widget::readfarme()
{
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
    if (frame.empty()) {
        qDebug() << "没打开摄像头";
        return;
    }
    frameCounter++; // 💖 每一帧都把计数器加1

       // 👇 ----- 这是我们新的判断逻辑！----- 👇
       if (frameCounter % processInterval == 0) // 如果当前帧数是处理间隔的倍数
       {
           // 到了该处理YOLO的帧啦！
           qDebug() << "🚀 Frame" << frameCounter << "- YOLO Processing!"; // 加个日志看看

           if (net.empty()) { // 再次检查模型是否加载
               qDebug() << "🚦 YOLO模型还没准备好，readfarme() 本轮YOLO处理跳过...";
               // 这里可以选择是否在跳过YOLO处理时清空上一帧的检测结果，
               // 或者保持上一帧的检测结果显示。目前我们只在处理时画框。
           } else {
               // -------- 🚀 YOLO 处理流程开始！🚀 --------
               cv::Mat blob;
               cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
               net.setInput(blob);
               std::vector<cv::Mat> outs;
               net.forward(outs, getOutputsNames(net)); // 执行前向传播
               postprocess(frame, outs); // ✨ 在这里画框 (postprocess会修改frame)
               // -------- 🚀 YOLO 处理流程结束！🚀 --------
           }
           // 如果 frameCounter 达到一个很大的值，可以考虑把它重置，避免溢出 (虽然int很大，但好习惯)
           // if (frameCounter >= 10000) { // 比如每10000帧重置
           //     frameCounter = 0;
           // }
       }
       particle_manager_.updateAndDrawParticles(frame);
    // 将处理后的图像显示到界面上
    imag = Mat2QImage(frame);
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
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // 绿色框框，粗细为2
        std::string text = label + ": " + cv::format("%.2f", confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int top_label = std::max(box.y, labelSize.height); // 确保文字不会画出图像顶端
        cv::putText(frame, text, cv::Point(box.x, top_label - 5), // 文字位置稍微调整一下
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        qDebug() << "Detected:" << QString::fromStdString(label)
                 << "Conf:" << confidences[idx]
                 << "Box:" << box.x << box.y << box.width << box.height;
        // 👇 ----- 在这里添加计算中心点和画红点的代码！----- 👇
               // 1. 计算检测框的中心点
               int centerX_box = box.x + box.width / 2;
               int centerY_box = box.y + box.height / 2;
               cv::Point center_of_box(centerX_box, centerY_box); // 创建一个 Point 对象

               // 2. 画一个小红圆点作为中心点标记
               int radius = 3; // 圆点的半径，可以调整大小
               cv::Scalar red_color(0, 0, 255); // BGR 顺序，所以红色是 (0, 0, 255)
               int thickness = -1; // -1 表示填充圆形，如果想画空心圆可以改成 1 或 2
               cv::circle(frame, center_of_box, radius, red_color, thickness);

               qDebug() << "🎯 Center point at: (" << centerX_box << "," << centerY_box << ")";
               // 👆 ----- 红点代码添加完毕！----- 👆
                if (particle_manager_.isEnabled() && (frameCounter % processInterval == 0)) {
                       qDebug() << "Condition met, calling generateParticles for center:" << centerX_box << centerY_box;
                       particle_manager_.generateParticles(center_of_box); // ✨ 为当前检测到的手生成粒子
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
