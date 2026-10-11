"""First-use training save preparation, shared with the public CLI."""
import threading
from .native_activity import native_worker_busy
from pvz_gym.runtime.save_template import prepare_training_save
from pvz_gym.env_file import get_game_executable, set_game_executable

def prepare_from_gui(ui):
    if native_worker_busy(ui) or getattr(ui, "_preparing_training_save", False):
        ui.log("WARN", "请先停止训练、评估或其他游戏操作，再准备存档。")
        return
    values = getattr(ui, "_runtime_param_vars", {})
    variable = values.get("game_executable")
    executable = variable.get().strip() if variable is not None else get_game_executable()
    if not executable:
        ui.log("WARN", "请先选择完整游戏目录中的 PlantsVsZombies.exe。")
        return
    ui._preparing_training_save = True
    def worker():
        try:
            path = prepare_training_save(executable)
            set_game_executable(executable)
            message, level = f"训练存档已准备：{path}。可以开始训练。", "PASS"
        except Exception as exc:
            message, level = f"准备训练存档失败：{exc}", "FAIL"
        def finish():
            ui._preparing_training_save = False
            ui.log(level, message)
        ui.root.after(0, finish)
    threading.Thread(target=worker, daemon=True).start()
