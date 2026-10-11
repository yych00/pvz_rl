"""Explicit real-PvZ acceptance using owned instances and generated saves.

python -m tests.integration.manual.public_acceptance --game-exe PATH
Writes ignored build/public-acceptance.json and runtime/runs artifacts.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from pvz_gym.env_file import repository_root
from pvz_gym.runtime import RuntimeConfig
from pvz_gym.runtime.save_template import prepare_training_save
from train.dqn.config import DoubleDQNConfig
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.evaluate import evaluate_checkpoint
from train.runtime import TrainingRuntimeConfig
from train.sampler import native_environment

def directory_hashes(root):
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-exe", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = repository_root()
    report = {"errors": [], "events": []}
    original = args.game_exe.resolve().parent
    personal = Path(os.environ.get("ProgramData", r"C:\ProgramData"))/"PopCap Games"/"PlantsVsZombies"/"userdata"
    before = {"game": directory_hashes(original), "personal": directory_hashes(personal)}
    template = prepare_training_save(args.game_exe)
    template_before = directory_hashes(template)
    def event(name, data):
        if name in ("error", "worker_recovery", "evaluation_error", "state"):
            report["events"].append({"event": name, "state": data.get("state"), "error": str(data.get("error", ""))})
            print(f"{name}: {data.get('state', '')} {data.get('error', '')}", flush=True)
    config = DoubleDQNConfig()
    config = replace(config, total_steps=args.steps, replay_capacity=512, batch_size=8,
                     learning_starts=16, train_every=8, auto_train_every=False,
                     autosave_every=max(32,args.steps//2), checkpoint_every=args.steps,
                     evaluation_mode="off", device=args.device,
                     env=replace(config.env,level_id=4,playthrough=1,simulation_mode="turbo",max_episode_ticks=3000,
                                 runtime=RuntimeConfig(executor="resident",transport="binary",snapshot="shared")),
                     runtime=TrainingRuntimeConfig(num_envs=1,game_executable=str(args.game_exe.resolve()),
                                                   instance_root=str(root/"runtime/acceptance/instances"),
                                                   save_template=str(template),worker_timeout_s=90))
    started=time.monotonic()
    try:
        # Verify generated profile is actually readable and can enter each scene.
        env, instance=native_environment(config,config.runtime,0,None,None)
        try:
            levels=[]
            for level,playthrough in ((4,1),(14,1),(24,2),(34,2),(44,2)):
                env.level_id=level; env.playthrough=playthrough
                _observation, info=env.reset(seed=1000+level)
                _observation, reward, terminated, truncated, step=env.step(0)
                deck = info.get("seed_selection", {}).get("actual", [])
                if not deck or step.get("ticks_advanced", 0) <= 0:
                    raise AssertionError(f"Scene {level}/{playthrough} did not select cards and advance")
                levels.append({"level":level,"playthrough":playthrough,
                               "ticks":step.get("ticks_advanced"),"reward":reward,
                               "deck":deck})
                print(f"scene {level}/{playthrough}: deck={deck}, ticks={step.get('ticks_advanced')}", flush=True)
            report["scene_probe"]=levels
        finally:
            try: env.close()
            finally: instance.close()
        for count in (1,2):
            run=root/"runs"/f"public-acceptance-{time.time_ns()}-{count}env"
            current=replace(config,runtime=replace(config.runtime,num_envs=count,asynchronous=count>1,inference_batch_size=count))
            trainer=DoubleDQNTrainerController(current,run_dir=run,on_event=event)
            summary=trainer.run()
            if trainer.state!="FINISHED" or summary.env_steps!=args.steps or trainer._agent.update_count<=0:
                raise AssertionError(f"Training did not finish with learner updates: {trainer.state}, {summary.snapshot()}")
            report[f"train_{count}env"]={"summary":summary.snapshot(),"updates":trainer._agent.update_count,"run":str(run)}
            if count==1:
                resume=replace(current,total_steps=args.steps+64)
                continuation=DoubleDQNTrainerController(resume,run_dir=run,checkpoint_path=trainer.latest_checkpoint,on_event=event)
                restored=continuation.run()
                if not restored.replay_restored or restored.env_steps!=args.steps+64:
                    raise AssertionError("Full checkpoint/Replay resume failed")
                report["resume"]={"summary":restored.snapshot(),"run":str(run)}
                evaluation=evaluate_checkpoint(continuation.latest_checkpoint,seed_start=10001,seed_count=2,
                                               max_episode_ticks=3000,simulation_mode="turbo",device=args.device)
                if evaluation.episodes!=2:
                    raise AssertionError("Fixed-seed evaluation incomplete")
                report["evaluation"]=asdict(evaluation)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["protected_unchanged"]=(before=={"game":directory_hashes(original),"personal":directory_hashes(personal)}
                                       and template_before==directory_hashes(template))
        report["wall_seconds"]=time.monotonic()-started
        (root/"build").mkdir(exist_ok=True)
        (root/"build/public-acceptance.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
        print(json.dumps(report,indent=2,default=str))
        if not report["protected_unchanged"]:
            raise AssertionError("Original game, personal saves or template changed")

if __name__ == "__main__":
    main()
