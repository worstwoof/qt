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
    , confThreshold(0.5f)
    , nmsThreshold(0.4f)
    , inpWidth(416)
    , inpHeight(416)
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
        qDebug() << "⏳ Trying to load YOLO model...";
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
        ifs.close(); // 好习惯，关掉它


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
}

void Widget::opencamara()
{
    capture.open(0);
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
    cv::Mat blob;
    // 1. 创建 blob: 把图像转换成YOLO网络能接受的格式
    //    参数：图像, 输出blob, 缩放因子(1/255归一化), 目标尺寸(inpWidth,inpHeight), 均值减法(这里是0), 是否交换RB通道(YOLO通常要BGR->RGB所以是true), 是否裁剪
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
    // 2. 设置网络输入
    net.setInput(blob);

    // 3. 前向传播：让网络进行预测！
    std::vector<cv::Mat> outs; // 存储网络所有输出层的结果
    net.forward(outs, getOutputsNames(net)); // 函数获取输出层名字

    // 4. 后处理：从网络输出中提取信息，画框框
    postprocess(frame, outs); // 使用我们另一个小帮手函数

    // 将处理后的图像显示到界面上
    imag = Mat2QImage(frame); // 用你已有的 Mat2QImage 函数
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




