"""Bounded hysteresis for asynchronous learner debt; earned updates stay intact."""


class TrainingCadence:
    def __init__(self, config, start_step=0, state=None):
        self.settings = self._settings(config)
        self.every = config.train_every
        self.last_change_step = start_step
        if isinstance(state, dict) and state.get('settings') == list(self.settings):
            every = state.get('every')
            changed = state.get('last_change_step')
            if (type(every) is int and 1 <= every <= self.maximum
                    and type(changed) is int and 0 <= changed <= start_step):
                self.every = every if config.auto_train_every else config.train_every
                self.last_change_step = changed

    @staticmethod
    def _settings(config):
        return (config.auto_train_every, config.train_every,
                config.auto_train_low_watermark, config.auto_train_high_watermark)

    @property
    def maximum(self):
        return self.settings[1] * 64

    def update(self, config, step, pending):
        settings = self._settings(config)
        if settings != self.settings:
            self.settings = settings
            self.every = config.train_every
            self.last_change_step = step
        previous = self.every
        if not config.auto_train_every:
            self.every = config.train_every
        elif step - self.last_change_step >= max(32, 2 * self.every):
            if pending >= config.auto_train_high_watermark:
                self.every = min(self.maximum, self.every * 2)
            elif pending <= config.auto_train_low_watermark:
                self.every = max(1, (self.every + 1) // 2)
            if self.every != previous:
                self.last_change_step = step
        return self.every != previous

    def snapshot(self):
        return dict(settings=list(self.settings), every=self.every,
                    last_change_step=self.last_change_step)
