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
    float reference_inner_radius;
    float reference_outer_radius;
     float particle_rotation_speed; // 每帧旋转的弧度s
    int inpWidth;
    int inpHeight;
    int frameCounter;
      int processInterval;
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net_param);
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
       struct Particle {
           cv::Point center;       // 粒子的中心点
           cv::Point base_center;  // 所属特效的中心点
           float initial_radius_ring; // 所属圆环的初始半径
           float angle_on_ring;    // 在圆环上的角度
           float current_radius_ball; // 小球当前的半径
           cv::Scalar color;       // 粒子的颜色
           int lifetime;           // 剩余生命周期
           int max_lifetime;       // 最大生命周期
           int ring_index;         // 所属的圆环层级 (0是最内圈)
           float expansion_speed;  // 圆环扩张速度
           float current_ring_radius_from_center;
       };
       std::vector<Particle> activeParticles; // 存储所有当前活跃的粒子
       void generateParticles(const cv::Point& effect_center);
       void updateAndDrawParticles(cv::Mat& frame);
bool particles_enabled; // 控制粒子特效是否开启的标志
       // 随机数生成器
       std::mt19937 rng_color_gen;
       std::uniform_int_distribution<int> dist_color_val;
       // 特效参数
       int particle_effect_duration; // 特效持续的总帧数
       int num_rings;                // 要生成的圆环数量
       int particles_per_ring;       // 每个圆环上的粒子数量
       float ring_initial_radius_step; // 每个圆环之间的初始半径差
       float ball_initial_radius;    // 小球的初始半径
       float ball_max_radius_factor; // 小球最大半径相对于初始半径的倍数
       float ring_expansion_rate;
};
#endif // WIDGET_H
