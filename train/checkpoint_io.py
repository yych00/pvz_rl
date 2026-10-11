"""Serialize owned checkpoint snapshots outside the sampling/learning loop."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import shutil
import tempfile
import uuid
import numpy as np


class DiskReplaySnapshot(dict):
    """Immutable file-backed arrays owned until the checkpoint write finishes."""
    def __init__(self, payload, directory):
        super().__init__()
        self._temporary = tempfile.TemporaryDirectory(prefix=".replay-snapshot-", dir=directory)
        self.directory = Path(self._temporary.name)
        try:
            for index, (name, value) in enumerate(payload.items()):
                if value.nbytes < 1024 * 1024:
                    self[name] = value.copy()
                    continue
                mapped = np.memmap(self.directory / f"{index}.dat", mode="w+",
                                   dtype=value.dtype, shape=value.shape)
                self[name] = mapped
                source, target = value.reshape(-1), mapped.reshape(-1)
                chunk = max(1, 16 * 1024 * 1024 // value.dtype.itemsize)
                for start in range(0, source.size, chunk):
                    target[start:start + chunk] = source[start:start + chunk]
                mapped.flush()
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in self.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        self.clear()
        self._temporary.cleanup()


def freeze_replay(payload, *, directory, spill_threshold=256 * 1024 * 1024):
    """Keep large asynchronous snapshots out of the process's commit charge."""
    if sum(value.nbytes for value in payload.values()) >= spill_threshold:
        return DiskReplaySnapshot(payload, directory)
    return {name: value.copy() for name, value in payload.items()}


def freeze_checkpoint(value):
    """Detach tensors and arrays before the learner is allowed to mutate them."""
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return {k: freeze_checkpoint(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(freeze_checkpoint(v) for v in value)
    if isinstance(value, list): return [freeze_checkpoint(v) for v in value]
    return copy.deepcopy(value)


class CheckpointWriter:
    """One pending save, with completion callbacks on the polling caller thread."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-checkpoint")
        self.pending = None
        self.on_complete = None

    def flush(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            callback, self.on_complete = self.on_complete, None
            destination, replay_file = pending.result()
            if callback:
                callback(destination, replay_file)
            return destination

    def poll(self):
        if self.pending is not None and self.pending.done():
            return self.flush()

    def submit(self, destination, model, replay, *, sync_latest=False, on_complete=None):
        try:
            self.flush()
            destination = Path(destination)
            from train.dqn.checkpoint_contract import declare_checkpoint
            declare_checkpoint(model, replay)
        except BaseException:
            if isinstance(replay, DiskReplaySnapshot):
                replay.close()
            raise
        # model/replay are already frozen by the controller at one decision boundary.
        def write():
            try:
                return write_files()
            finally:
                if isinstance(replay, DiskReplaySnapshot):
                    replay.close()

        def write_files():
            from train.dqn.agent import _atomic_torch_save
            from train.dqn.replay import ReplayBuffer
            destination.parent.mkdir(parents=True, exist_ok=True)
            replay_file = destination.with_name(f"{destination.stem}-replay-{model['env_step']:09d}-{uuid.uuid4().hex}.npz")
            if replay is not None:
                ReplayBuffer.write_checkpoint_payload(replay_file, replay)
                model["replay_file"] = replay_file.name
            _atomic_torch_save(model, destination)
            alias = destination.parent / ("latest-full.pt" if replay is not None else "latest-model.pt")
            if alias != destination:
                temporary_alias = alias.with_suffix(".pt.tmp")
                try:
                    shutil.copy2(destination, temporary_alias)
                    temporary_alias.replace(alias)
                finally:
                    temporary_alias.unlink(missing_ok=True)
            if sync_latest and destination.name != "latest.pt":
                latest = destination.parent / "latest.pt"
                temporary = latest.with_suffix(".pt.tmp")
                try: shutil.copy2(destination, temporary); temporary.replace(latest)
                finally: temporary.unlink(missing_ok=True)
            return destination, replay_file
        self.on_complete = on_complete
        try:
            self.pending = self.executor.submit(write)
        except BaseException:
            if isinstance(replay, DiskReplaySnapshot):
                replay.close()
            raise

    def close(self):
        try: self.flush()
        finally: self.executor.shutdown(wait=True)
