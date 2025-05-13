#ifndef PRACTICLESYSTEM_H
#define PRACTICLESYSTEM_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <random>
#include <QColor>
using namespace std;


struct Particle {
    cv::Point center;       // 粒子（小圆球）的中心点
    cv::Point base_center;  // 所属特效的中心点 (手掌中心)
    float initial_radius_ring; // 所属圆环的初始半径
    float angle_on_ring;    // 在圆环上的角度
    float current_radius_ball; // 小球当前的半径
    cv::Scalar color;       // 粒子的颜色
    int lifetime;           // 剩余生命周期 (单位：帧)
    int max_lifetime;       // 最大生命周期
    int ring_index;         // 所属的圆环层级 (0是最内圈)
    float expansion_speed;  // 圆环扩张速度（可选）
    float current_ring_radius_from_center;
};

class practiclesystem
{
public:
    practiclesystem();
    void generateParticles(const cv::Point& effect_center);
    void updateAndDrawParticles(cv::Mat& frame);
    void setEnabled(bool enabled);
    bool isEnabled() const;
    void clearActiveParticles();
    bool enabled_;
    practiclesystem(int effect_duration);
private:
    int particle_effect_duration; // 粒子持续的总帧数
    int num_rings;                // 要生成的圆环数量
    int particles_per_ring;       // 每个圆环上的粒子数量
    float ring_initial_radius_step; // 每个圆环之间的初始半径差
    float ball_initial_radius;    // 小球的初始半径
    float ball_max_radius_factor; // 小球最大半径相对于初始半径的倍数 (用于从小变大再变小)
    float ring_expansion_rate;    // 圆环向外扩张的速度 (每帧增加的半径)
    float particle_rotation_speed; // 每帧旋转的弧度s
    float reference_inner_radius;
    float reference_outer_radius;
    std::mt19937 rng_engine_;
    std::uniform_int_distribution<int> dist_color_channel_val_;
 std::vector<Particle> activeParticles; // 存储所有当前活跃的粒子
};
#endif // PRACTICLESYSTEM_H
