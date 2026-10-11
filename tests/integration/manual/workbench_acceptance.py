"""Functional GUI acceptance with a real owned game and CUDA model.

Runs the Workbench callbacks in Tk's event loop; it does not inject OS input.
Checks actual Q-overlay GDI drawing. Screen appearance still needs visual review.
"""
import argparse
import json
import os
from pathlib import Path
import time
import tkinter as tk
import torch

import workbench.main  # installs the same GUI patches as the normal launcher
from workbench import PvzGymUnitTestGUI
from workbench.gui.game_setup import prepare_from_gui
from workbench.gui.main_window import request_workbench_close
from workbench.gui import q_heatmap_overlay as overlay_module
from pvz_gym.env_file import repository_root
from tests.integration.manual.public_acceptance import directory_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-exe", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch and an available NVIDIA GPU are required for this acceptance")
    torch.set_num_threads(1)
    os.environ["PVZ_GAME_EXECUTABLE"] = str(args.game_exe.resolve())
    repo = repository_root()
    personal = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "PopCap Games/PlantsVsZombies/userdata"
    protected = (args.game_exe.resolve().parent, personal, repo / "local/save_template")
    before = [directory_hashes(p) for p in protected]
    report = {"device": "cuda", "q_frames": 0, "successful_gdi_draws": 0, "errors": [], "logs": []}
    root = tk.Tk()
    app = PvzGymUnitTestGUI(root)
    app._switch_left_tab("training")
    app._use_visible_training_preset()
    app.var_training_q_overlay.set("Top 5 热力图")
    app.var_training_max_episode_ticks.set("3000")
    for key, value in {"total_steps": "64", "replay_capacity": "512", "batch_size": "8",
                       "learning_starts": "16", "train_every": "8", "auto_train_every": "False",
                       "autosave_every": "32", "checkpoint_every": "64", "evaluation_mode": "off",
                       "device": "cuda"}.items():
        app._training_param_vars[key].set(value)
    original_log = app.log
    def log(level, message):
        report["logs"].append({"level": level, "message": message})
        original_log(level, message)
    app.log = log
    overlay = overlay_module.get_q_heatmap_overlay()
    original_frame = overlay.update_frame
    def frame(data):
        report["q_frames"] += 1
        report["last_q"] = {key: data.get(key) for key in ("process_id", "noop_q", "top5", "sequence_id")}
        return original_frame(data)
    overlay.update_frame = frame
    original_draw = overlay_module.gdi32.BitBlt
    def draw(*args):
        result = original_draw(*args)
        if result:
            report["successful_gdi_draws"] += 1
        return result
    overlay_module.gdi32.BitBlt = draw
    started = time.monotonic()
    controller = None
    launched = False
    def poll():
        nonlocal launched, controller
        if time.monotonic() - started > 240:
            report["errors"].append("Workbench acceptance exceeded 240 seconds")
            request_workbench_close(app)
            return
        if not launched:
            if getattr(app, "_preparing_training_save", False):
                root.after(100, poll)
                return
            report["prepare_success"] = any("训练存档已准备" in item["message"] for item in report["logs"])
            if not report["prepare_success"]:
                report["errors"].append("GUI save preparation failed")
                request_workbench_close(app)
                return
            app._start_training()
            controller = app._training_controller
            launched = True
            if controller is None:
                report["errors"].append("Workbench did not create its trainer")
                request_workbench_close(app)
                return
        if not app._training_running:
            report["state"] = controller.state
            report["steps"] = controller.summary.env_steps
            report["updates"] = controller._agent.update_count
            report["model_device"] = str(controller._agent.device)
            report["run"] = str(controller.run_dir)
            if controller.state != "FINISHED" or report["steps"] != 64 or report["updates"] <= 0:
                report["errors"].append("GUI training did not complete")
            if report["q_frames"] == 0 or report["successful_gdi_draws"] == 0:
                report["errors"].append("Q overlay did not receive and draw real model frames")
            request_workbench_close(app)
            return
        root.after(100, poll)
    try:
        root.after(100, lambda: prepare_from_gui(app))
        root.after(200, poll)
        root.mainloop()
    finally:
        overlay.stop()
        overlay.update_frame = original_frame
        overlay_module.gdi32.BitBlt = original_draw
        report["recoveries"] = sum("[TRAIN:RECOVERY]" in item["message"] for item in report["logs"])
        report["protected_unchanged"] = before == [directory_hashes(p) for p in protected]
        if not report["protected_unchanged"]:
            report["errors"].append("Original game, personal saves or save template changed")
        report["wall_seconds"] = time.monotonic() - started
        (repo / "build/workbench-acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "logs"}, indent=2, default=str))
    if report["errors"]:
        raise RuntimeError(report["errors"])


if __name__ == "__main__":
    main()
