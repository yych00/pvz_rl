# -*- coding: utf-8 -*-
"""Shared Dialog cleanup policy for Gym/unit-test entry points."""


def install_dialog_cleanup_policy(client_cls) -> None:
    """
    Extend the existing target-dialog cleanup set with DIALOG_CONTINUE (37).

    The existing client.select_level()/select_level_id() paths already call
    cleanup_target_dialogs() before and after a level switch, and scene-change
    cleanup delegates to the same helper.  Keeping CONTINUE in this shared set
    means single-level LIVE tests, GUI level jumps, and transition cleanup all
    use exactly the same policy instead of adding a second polling loop.
    """
    client_cls.DIALOG_CONTINUE = 37

    current = tuple(getattr(client_cls, "TARGET_SCENE_DIALOGS", (2, 19, 17)))
    if client_cls.DIALOG_CONTINUE not in current:
        client_cls.TARGET_SCENE_DIALOGS = current + (client_cls.DIALOG_CONTINUE,)

    names = dict(getattr(client_cls, "DIALOG_NAMES", {}))
    names[client_cls.DIALOG_CONTINUE] = "DIALOG_CONTINUE(37/继续游戏)"
    client_cls.DIALOG_NAMES = names
