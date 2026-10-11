"""Explicit checkpoint contents and restore semantics shared by every entrypoint."""

CHECKPOINT_LOAD_RESUME = "resume"  # Compatibility: matching replay, otherwise reported rebuild.
CHECKPOINT_LOAD_STRICT = "strict_resume"
CHECKPOINT_LOAD_MODEL = "model_resume"
CHECKPOINT_LOAD_WEIGHTS_ONLY = "weights_only"
CHECKPOINT_LOAD_WEIGHTS_REPLAY = "weights_replay"
CHECKPOINT_LOAD_MODES = (CHECKPOINT_LOAD_RESUME, CHECKPOINT_LOAD_STRICT,
                         CHECKPOINT_LOAD_MODEL, CHECKPOINT_LOAD_WEIGHTS_ONLY,
                         CHECKPOINT_LOAD_WEIGHTS_REPLAY)


def declare_checkpoint(model, replay=None):
    """Annotate the frozen snapshot; full means training state, never live Board."""
    full = replay is not None
    model.update(checkpoint_format_version=1, checkpoint_kind="full" if full else "model_only",
                 strict_resume_supported=full, exact_game_resume=False,
                 replay_snapshot_step=int(model["env_step"]) if full else None,
                 replay_snapshot_episode=int(model["episode"]) if full else None)
    if not full:
        model.pop("replay_file", None)
    return model


def inspect_checkpoint(model):
    kind = model.get("checkpoint_kind", "legacy_unknown")
    if kind not in ("full", "model_only", "legacy_unknown"):
        raise ValueError(f"unsupported checkpoint_kind: {kind!r}")
    version = model.get("checkpoint_format_version", 1)
    if version != 1:
        raise ValueError(f"unsupported checkpoint format version: {version!r}")
    if kind == "full":
        if not model.get("replay_file") or not model.get("strict_resume_supported"):
            raise ValueError("full checkpoint is missing its replay pairing")
        for key, expected in (("replay_snapshot_step", model["env_step"]),
                              ("replay_snapshot_episode", model["episode"])):
            if model.get(key) != expected:
                raise ValueError("full checkpoint replay metadata does not match the training point")
    return kind
