#include "practiclesystem.h"
#include <QDebug>


practiclesystem::practiclesystem(int effect_duration)
    : particle_effect_duration(effect_duration),
      enabled_(false) // 默认关闭，或者你之前的默认值
       , num_rings(3)   // ✨ 以下都是从 Widget 构造函数移过来的初始化 ✨
       , particles_per_ring(12)
       , ring_initial_radius_step(30.0f)
       , ball_initial_radius(3.0f)
       , ball_max_radius_factor(2.0f)
       , ring_expansion_rate(0.5f)
       , particle_rotation_speed(0.03f)
       , rng_color_gen(std::random_device()())
       , dist_color_val(0, 255)
       // reference_inner_radius_ 和 reference_outer_radius_ 的初始化
   {
       reference_inner_radius = ring_initial_radius_step;
       reference_outer_radius = (num_rings* ring_initial_radius_step) + 100.0f;
   }

void practiclesystem::setEnabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled_) {
        activeParticles.clear(); // 如果关闭，清空粒子
    }
}

bool practiclesystem::isEnabled() const {
    return enabled_;
}

void practiclesystem::clearActiveParticles() {
    activeParticles.clear();
}
void practiclesystem::generateParticles(const cv::Point& effect_center) {
    // activeParticles.clear(); // 🎉 确保这行被注释掉或删除了！ 🎉
    qDebug() << "Adding particles for effect_center:" << effect_center.x << effect_center.y << "Current active count:" << activeParticles.size();

    for (int r_idx = 0; r_idx < num_rings; ++r_idx) {
        float current_ring_initial_radius = (r_idx + 1) * ring_initial_radius_step;
        for (int p_idx = 0; p_idx < particles_per_ring; ++p_idx) {
            Particle p;
            p.base_center = effect_center; // 粒子记住它是由哪个手掌中心触发的
            p.initial_radius_ring = current_ring_initial_radius;
            p.angle_on_ring = (2.0 * CV_PI / particles_per_ring) * p_idx;

            // 初始化 current_ring_radius_from_center
            p.current_ring_radius_from_center = p.initial_radius_ring;

            p.current_radius_ball = ball_initial_radius; // 初始小球半径
            p.color = cv::Scalar(dist_color_val(rng_color_gen), dist_color_val(rng_color_gen), dist_color_val(rng_color_gen));
            p.max_lifetime = particle_effect_duration;
            p.lifetime = p.max_lifetime;
            p.ring_index = r_idx;
            p.expansion_speed = this->ring_expansion_rate; // 或者粒子特定的速度

            activeParticles.push_back(p);
        }
    }
    qDebug() << "After adding, new active count:" << activeParticles.size();
}
void practiclesystem::updateAndDrawParticles(cv::Mat& frame) {
    qDebug() << "updateAndDrawParticles called. Active particles:" << activeParticles.size()
                << "Particles enabled:" << enabled_; // 也打印一下 enabled 状态
    if (activeParticles.empty()) return;

    std::vector<Particle> next_gen_particles;

    for (Particle& p : activeParticles) {
        p.lifetime--; // 生命减少

        if (p.lifetime > 0) {
            // float life_progress = 1.0f - (static_cast<float>(p.lifetime) / p.max_lifetime); // 如果不再需要，可以注释掉

            // 1. 圆环扩张 (每帧固定速度增加)
            //    (确保 p.current_ring_radius_from_center 在 generateParticles 中被正确初始化为 p.initial_radius_ring)
            //    (确保 p.expansion_speed 在 generateParticles 中被正确初始化，例如 p.expansion_speed = this->ring_expansion_rate;)
            p.current_ring_radius_from_center += p.expansion_speed;
            // ✨✨ 2. 更新粒子的角度 (实现旋转) ✨✨
                   p.angle_on_ring += this->particle_rotation_speed; // 或者 p.angular_velocity 如果每个粒子速度不同
                   // 确保角度在 0 到 2*PI 之间 (可选，但有助于避免数值过大)
                   if (p.angle_on_ring > 2.0 * CV_PI) {
                       p.angle_on_ring -= 2.0 * CV_PI;
                   } else if (p.angle_on_ring < 0.0f) { // 如果速度是负的，可能需要这个
                       p.angle_on_ring += 2.0 * CV_PI;
                   }

            // 重新计算粒子在扩张圆环上的位置
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

