"""Collapsible parameter groups; visibility never changes parameter values."""

import tkinter as tk


class ParameterSection(tk.Frame):
    def __init__(self, parent, ui, title):
        super().__init__(parent, bg=ui.CLR_CARD)
        self.title = title
        self.expanded = False
        self._fields = []
        self.button = tk.Button(
            self, command=self.toggle, anchor="w", relief="flat", bd=0,
            font=ui.FONT_BOLD, bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MUTED,
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
            cursor="hand2", padx=9, pady=8,
        )
        self.button.pack(fill=tk.X)
        self.body = tk.Frame(self, bg=ui.CLR_CARD)
        for col in range(3):
            self.body.columnconfigure(col, weight=1, uniform="advanced-parameter")
        self.bind("<Destroy>", self._cleanup, add="+")
        self._refresh()

    def watch(self, variable, default):
        trace = variable.trace_add("write", lambda *_: self._refresh())
        self._fields.append((variable, str(default), trace))
        self._refresh()

    def _refresh(self):
        changed = sum(str(var.get()) != default for var, default, _ in self._fields)
        status = f"已修改 {changed} 项" if changed else "默认配置"
        self.button.config(text=f"{'▼' if self.expanded else '▶'} {self.title} · {status}")

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill=tk.X, pady=(8, 0))
        else:
            self.body.pack_forget()
        self._refresh()

    def _cleanup(self, event):
        if event.widget is self:
            for variable, _, trace in self._fields:
                variable.trace_remove("write", trace)
