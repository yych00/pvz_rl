#include "action_mask.h"

#include "action_validator.h"
#include "../game/original/cob_calls.h"
#include "../game/original/plant_calls.h"
#include "../game/original/shovel_calls.h"
#include "../game/original/vase_calls.h"

namespace pvz::action {

void ActionMask::compute_mask(std::vector<bool>& out_mask) {
    out_mask.assign(kActionSpaceSize, false);

    // 空操作仍走统一验证器。
    Action action;
    if (ActionCodec::decode(ActionCodec::kNoOpIndex, action)) {
        out_mask[ActionCodec::kNoOpIndex] = ActionValidator::can_execute(action, nullptr);
    }

    // 种植保留双路径校验：原版 Board::CanPlantAt 批量提供空间判断，
    // 手写验证器用于普通关卡和差异统计。冒险 1-5/1-10 以及坚果
    // 保龄球挑战采用原版空间结果，卡包可用性由传送带状态判断。
    std::vector<bool> plant_mask;
    if (pvz::game::original::PlantCalls::build_flat_action_mask_shadow(plant_mask) &&
        static_cast<int>(plant_mask.size()) == ActionCodec::kPlantCount) {
        for (int offset = 0; offset < ActionCodec::kPlantCount; ++offset) {
            out_mask[ActionCodec::kPlantBase + offset] =
                plant_mask[static_cast<std::size_t>(offset)];
        }
    } else {
        // 防御性后备路径。正常已附加战斗中，即使原版候选路径不可用，
        // 影子构建器仍会返回旧验证器结果；这里主要处理进程或生命周期异常。
        for (int offset = 0; offset < ActionCodec::kPlantCount; ++offset) {
            const int index = ActionCodec::kPlantBase + offset;
            Action fallback;
            if (!ActionCodec::decode(index, fallback)) continue;
            out_mask[index] = ActionValidator::can_execute(fallback, nullptr);
        }
    }

    // 一次扫描同时生成普通铲和南瓜铲掩码，执行时仍重新解析目标。
    std::vector<bool> shovel_mask;
    if (pvz::game::original::ShovelCalls::build_flat_action_mask(shovel_mask) &&
        static_cast<int>(shovel_mask.size()) == ActionCodec::kShovelCount) {
        for (int offset = 0; offset < ActionCodec::kShovelCount; ++offset) {
            out_mask[ActionCodec::kShovelBase + offset] =
                shovel_mask[static_cast<std::size_t>(offset)];
        }
    }

    // 玉米炮只按落点编码；一次扫描决定是否存在就绪炮。
    std::vector<bool> cob_mask;
    if (pvz::game::original::CobCalls::build_flat_action_mask(cob_mask) &&
        static_cast<int>(cob_mask.size()) == ActionCodec::kCobCount) {
        for (int offset = 0; offset < ActionCodec::kCobCount; ++offset) {
            out_mask[ActionCodec::kCobBase + offset] =
                cob_mask[static_cast<std::size_t>(offset)];
        }
    }
    // INTERACT_CELL currently supports Vasebreaker only. Whack-a-Zombie
    // hammer targeting is not implemented and remains masked false.
    std::vector<bool> vase_mask;
    if (game::original::VaseCalls::build_flat_action_mask(vase_mask) && vase_mask.size() == 54)
        for (int cell = 0; cell < 54; ++cell)
            out_mask[ActionCodec::kInteractBase + cell] = vase_mask[static_cast<std::size_t>(cell)];
}

}  // 命名空间 pvz::action
