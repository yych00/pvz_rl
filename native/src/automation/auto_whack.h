#pragma once

namespace pvz::automation {

class AutoWhack {
public:
    // 只在冒险 2-5 或“锤僵尸”小游戏中发送一次锤击。
    static bool whack_one();

    // 在冒险 2-5 或“锤僵尸”小游戏中对所有场上合格僵尸执行批量连续锤击。
    // 返回本次成功发送锤击的目标数量。
    static int whack_all();
};

}  // namespace pvz::automation
