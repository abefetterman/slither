

class LinearSchedule:
    def __init__(self, timesteps, initial, final):
        self.timesteps = timesteps
        self.initial = initial
        self.final = final
    def at(self, t):
        f = min(float(t) / self.timesteps, 1.0)
        return self.initial + f*(self.final - self.initial)


class ExpoSchedule:
    def __init__(self, steps, initial, factor):
        self.steps = steps
        self.initial = initial
        self.factor = factor
    def at(self, t):
        return self.initial*(self.factor ** (float(t) / self.steps))
