#pragma once

#include <cstdint>
#include <string>

namespace pvz::control {

// 训练受控实验注入 Hook
class TrainingHooks {
public:
    // 设置游戏随机数生成器 (RNG) 种子；返回当前版本是否原生支持
    static bool set_rng_seed(std::uint64_t seed);

    // 受控刷新指定类型僵尸于指定行
    static bool spawn_zombie(int zombie_type, int row);

    // 注入自定义波次配置 JSON
    static bool set_wave_config(const std::string& config_json);

    // 强制触发阳光掉落
    static bool spawn_sun(int value, float x, float y);
};

}  // namespace pvz::control
