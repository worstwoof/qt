#include "practiclesystem.h"
#include <QDebug>


practiclesystem::practiclesystem(int effect_duration)
    : particle_effect_duration(effect_duration*1.2),
      enabled_(false) // 默认关闭，或者你之前的默认值
       , num_rings(4)
       , particles_per_ring(12)
       , ring_initial_radius_step(17.0f)
       , ball_initial_radius(2.3f)
       , ball_max_radius_factor(2.0f)
       , ring_expansion_rate(0.5f)
       , particle_rotation_speed(0.04f),
    rng_engine_(std::random_device()()),
        dist_color_channel_val_(0, 255)


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
    // activeParticles.clear(); //
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
            int r_val = rand() % 255;
               int g_val = rand() % 255;
               int b_val = rand() % 255;
p.color = cv::Scalar(b_val, g_val, r_val);
                   activeParticles.push_back(p);
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

            p.current_ring_radius_from_center += p.expansion_speed;

                   p.angle_on_ring += this->particle_rotation_speed;

                   if (p.angle_on_ring > 2.0 * CV_PI) {
                       p.angle_on_ring -= 2.0 * CV_PI;
                   } else if (p.angle_on_ring < 0.0f) {
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


                cv::circle(frame, p.center, static_cast<int>(p.current_radius_ball), p.color, -1);


            next_gen_particles.push_back(p);
        }
    }
    activeParticles = next_gen_particles;
}

