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
  , particle_rotation_speed(0.03f) // 正数逆时针，负数顺时针。值越大转得越快。

    , nmsThreshold(0.4f)

    , inpWidth(416)
    , inpHeight(416)

    , frameCounter(0)
       , processInterval(8)
    , rng_color_gen(std::random_device{}())
       , dist_color_val(0, 255)
       , particle_effect_duration(8)
       , num_rings(4)                         // 生成4个同心圆环
       , particles_per_ring(12)                //  每个圆环12个粒子
       , ring_initial_radius_step(30.0f)      //  圆环间初始半径差 20 像素
       , ball_initial_radius(4.0f)            // 小球初始半径 3 像素
       , ball_max_radius_factor(2.0f)         // 小球会先变大到初始的2倍，然后消失
  , reference_inner_radius(ring_initial_radius_step)
  , reference_outer_radius( (num_rings * ring_initial_radius_step) + 150.0f )
       , ring_expansion_rate(0.5f)            //  圆环每帧向外扩张0.5像素
   , particles_enabled(false) // 初始化粒子特效为关闭状态

{
    ui->setupUi(this);
    this->setWindowFlag(Qt::FramelessWindowHint);
    this->installEventFilter(this);
 particle_effect_duration = processInterval; // 确保特效持续时间和处理间隔一致
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

void Widget::generateParticles(const cv::Point& effect_center) {

    qDebug() << "Adding particles for effect_center:" << effect_center.x << effect_center.y << "Current active count:" << activeParticles.size();

    for (int r_idx = 0; r_idx < num_rings; ++r_idx) {
        float current_ring_initial_radius = (r_idx + 1) * ring_initial_radius_step;
        for (int p_idx = 0; p_idx < particles_per_ring; ++p_idx) {
            Particle p;
            p.base_center = effect_center;
            p.initial_radius_ring = current_ring_initial_radius;
            p.angle_on_ring = (2.0 * CV_PI / particles_per_ring) * p_idx;

            // 初始化 current_ring_radius_from_center
            p.current_ring_radius_from_center = p.initial_radius_ring;

            p.current_radius_ball = ball_initial_radius;
            p.color = cv::Scalar(dist_color_val(rng_color_gen), dist_color_val(rng_color_gen), dist_color_val(rng_color_gen));
            p.max_lifetime = particle_effect_duration;
            p.lifetime = p.max_lifetime;
            p.ring_index = r_idx;
            p.expansion_speed = this->ring_expansion_rate;

            activeParticles.push_back(p);
        }
    }
    qDebug() << "After adding, new active count:" << activeParticles.size();
}
void Widget::updateAndDrawParticles(cv::Mat& frame) {
    qDebug() << "updateAndDrawParticles called. Active particles:" << activeParticles.size()
                << "Particles enabled:" << particles_enabled;
    if (activeParticles.empty()) return;

    std::vector<Particle> next_gen_particles;

    for (Particle& p : activeParticles) {
        p.lifetime--;

        if (p.lifetime > 0) {

            // 1. 圆环扩张 (每帧固定速度增加)

            p.current_ring_radius_from_center += p.expansion_speed;
            //  2. 更新粒子的角度 (实现旋转)
                   p.angle_on_ring += this->particle_rotation_speed; // 或者 p.angular_velocity 如果每个粒子速度不同

                   if (p.angle_on_ring > 2.0 * CV_PI) {
                       p.angle_on_ring -= 2.0 * CV_PI;
                   } else if (p.angle_on_ring < 0.0f) {
                       p.angle_on_ring += 2.0 * CV_PI;
                   }
            p.center.x = static_cast<int>(p.base_center.x + p.current_ring_radius_from_center * std::cos(p.angle_on_ring));
            p.center.y = static_cast<int>(p.base_center.y + p.current_ring_radius_from_center * std::sin(p.angle_on_ring));

            // 2. 根据距离中心的位置，线性计算小球半径
            float max_ball_radius_for_effect = this->ball_initial_radius * this->ball_max_radius_factor;
            float min_ball_radius_for_effect = this->ball_initial_radius * 0.3f; // 可调整

            float distance_from_center = p.current_ring_radius_from_center;

            if (distance_from_center <= this->reference_inner_radius) {
                p.current_radius_ball = max_ball_radius_for_effect;
            } else if (distance_from_center >= this->reference_outer_radius) {
                p.current_radius_ball = min_ball_radius_for_effect;
            } else {
                float t = (distance_from_center - this->reference_inner_radius) / (this->reference_outer_radius - this->reference_inner_radius);
                t = std::max(0.0f, std::min(1.0f, t));
                p.current_radius_ball = max_ball_radius_for_effect * (1.0f - t) + min_ball_radius_for_effect * t;
            }

            p.current_radius_ball = std::max(0.5f, p.current_radius_ball);

            // 绘制粒子
            if (p.current_radius_ball >= 0.5f) {
                cv::circle(frame, p.center, static_cast<int>(p.current_radius_ball), p.color, -1);
            }

            next_gen_particles.push_back(p);
        }
    }
    activeParticles = next_gen_particles;
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
    frameCounter++;

       if (frameCounter % processInterval == 0)
       {

           qDebug() << "Frame" << frameCounter << "- YOLO Processing!";

           if (net.empty()) { // 再次检查模型是否加载
               qDebug() << "YOLO模型还没准备好，readfarme() 本轮YOLO处理跳过...";
           } else {
               cv::Mat blob;
               cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
               net.setInput(blob);
               std::vector<cv::Mat> outs;
               net.forward(outs, getOutputsNames(net));
               postprocess(frame, outs); // 在这里画框 (postprocess会修改frame)
           }

           if (frameCounter >= 10000) {
             frameCounter = 0;
            }
       }
       updateAndDrawParticles(frame);
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

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
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
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // 绿色框，粗细为2
        std::string text = label + ": " + cv::format("%.2f", confidences[idx]);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int top_label = std::max(box.y, labelSize.height); // 确保文字不会画出图像顶端
        cv::putText(frame, text, cv::Point(box.x, top_label - 5), // 文字位置稍微调整一下
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        qDebug() << "Detected:" << QString::fromStdString(label)
                 << "Conf:" << confidences[idx]
                 << "Box:" << box.x << box.y << box.width << box.height;
               int centerX_box = box.x + box.width / 2;
               int centerY_box = box.y + box.height / 2;
               cv::Point center_of_box(centerX_box, centerY_box); // 创建一个 Point 对象
               int radius = 3; // 圆点的半径
               cv::Scalar red_color(0, 0, 255);
               int thickness = -1;
               cv::circle(frame, center_of_box, radius, red_color, thickness);

               qDebug() << " Center point at: (" << centerX_box << "," << centerY_box << ")";

                if (particles_enabled && (frameCounter % processInterval == 0)) {
                       qDebug() << "Condition met, calling generateParticles for center:" << centerX_box << centerY_box;
                       generateParticles(center_of_box); // 为当前检测到的手生成粒子
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
    particles_enabled = !particles_enabled;
    if (particles_enabled) {
        qDebug() << "Particle effects have been set to ENABLED by button click.";
        ui->pushButton_lizi->setText("关闭粒子特效");
    } else {
        qDebug() << "Particle effects have been set to DISABLED by button click. Clearing active particles.";
        activeParticles.clear();
        ui->pushButton_lizi->setText("打开粒子特效");
    }
}
