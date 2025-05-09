#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QMainWindow>
#include <QDebug>
#include <QTimer>
#include <QImage>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <random> // 💖 新增：用于生成随机颜色
#include "practiclesystem.h"

using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QLabel
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();


    bool eventFilter(QObject *watched,QEvent *event) override;

    QImage Mat2QImage(Mat cvImg);
    void on_pushButton_open_clicked();
 void on_pushButton_lizi_clicked(); // ✨ 新增：处理粒子特效按钮点击的槽函数
private slots:
    void opencamara();
    void readfarme();
    void closecamara();

private:
    Ui::Widget *ui;
    QTimer* timer;
    QImage imag;
    Mat cap,cap_gray,cat_tmp;
    VideoCapture capture;
    vector<Rect> handRect;
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confThreshold;
    float nmsThreshold;
   std::vector<std::string> getOutputsNames(const cv::dnn::Net& net_param);

    int inpWidth;
    int inpHeight;
    int frameCounter;       // 💖 新增：帧计数器，记录当前是第几帧啦
      int processInterval;    // 💖 新增：处理间隔，比如我们想每隔 processInterval 帧处理一次

    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);



      practiclesystem particle_manager_;





};
#endif // WIDGET_H
