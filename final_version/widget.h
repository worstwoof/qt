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
#include <QFileDialog>
#include <QDateTime>


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
    bool is_recording_;
    cv::VideoWriter video_writer_;
    QString video_output_path_;
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    bool camera_is_opened_;
    practiclesystem particle_manager_;
};
#endif // WIDGET_H
