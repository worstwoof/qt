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
#include <random>
#include "practiclesystem.h"
#include <QFileDialog> // ✨ 新增：用于让用户选择保存路径
#include <QDateTime>   // ✨ 新增：用于生成唯一的文件名


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
    void on_pushButton_camera_clicked();
    void on_pushButton_lizi_clicked();
    void on_pushButton_snapshot_clicked();
    void on_pushButton_record_clicked();
private slots:
    void readfarme();
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
    int frameCounter;
    int processInterval;
    bool is_recording_;                     // 是否正在录像的标志
    cv::VideoWriter video_writer_;          // OpenCV 的视频写入对象
    QString video_output_path_;             // 当前录像文件的保存路径
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    bool camera_is_opened_; // 跟踪摄像头状态
    practiclesystem particle_manager_;
};
#endif // WIDGET_H
