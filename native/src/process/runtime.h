#pragma once

namespace pvz::process {

bool attach();
bool game_on();
int ui();
int select_level(int level_id);
// Gym 测试和训练使用的单关重玩模式：
//   playthrough=1：模拟首次冒险流程（mFinishedAdventure=0）
//   playthrough=2：模拟通关后的重玩流程（mFinishedAdventure=1）
// 这是单局配置，不代表要推进整个存档流程。
int set_adventure_playthrough(int playthrough);
int current_level();

}  // 命名空间 pvz::process
