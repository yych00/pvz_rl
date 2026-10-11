# -*- coding: utf-8 -*-
"""Constructor-order hotfix for the integrated LIVE level matrix.

The main gui builds the level rows before the selected/current status labels are
created.  Rendering those rows accesses ``lbl_live_selected`` and therefore
crashes during GUI construction.  This patch defers the first row render until
``_build_live_panel`` has finished creating all dependent widgets.

The compact status label to the right of ``已选择 x-x`` duplicates the
right-hand console output.  Keep the widget alive because existing runtime code
updates it, but remove it from layout so LIVE status has one visible output
surface instead of two.
"""

from __future__ import annotations

from typing import Any


def install_live_gui_constructor_patch(gui_cls: Any) -> None:
    if getattr(gui_cls, "_live_gui_constructor_patch_installed", False):
        return

    original_build_live_panel = gui_cls._build_live_panel
    original_render_live_rows = gui_cls._render_live_rows

    def _render_live_rows_safe(self):
        if getattr(self, "_live_panel_initializing", False):
            return None
        if not hasattr(self, "lbl_live_selected") or not hasattr(self, "lbl_live_summary"):
            return None
        return original_render_live_rows(self)

    def _hide_duplicate_live_status(self):
        label = getattr(self, "lbl_live_current", None)
        if label is None:
            return
        manager = str(label.winfo_manager() or "")
        if manager == "grid":
            label.grid_remove()
        elif manager == "pack":
            label.pack_forget()
        elif manager == "place":
            label.place_forget()

    def _build_live_panel_safe(self, parent):
        self._live_panel_initializing = True
        try:
            result = original_build_live_panel(self, parent)
        finally:
            self._live_panel_initializing = False

        # ``lbl_live_current`` mirrors the right-side console, so keep it only as
        # an internal compatibility target for existing .config() calls.
        _hide_duplicate_live_status(self)

        # Now all labels/buttons created later in _build_live_panel exist, so the
        # first real render is safe and the selected row/status are initialized.
        original_render_live_rows(self)
        return result

    gui_cls._render_live_rows = _render_live_rows_safe
    gui_cls._build_live_panel = _build_live_panel_safe
    gui_cls._live_gui_constructor_patch_installed = True
