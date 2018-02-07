

class LinearSchedule:
    def __init__(self, timesteps, initial, final):
        self.timesteps = timesteps
        self.initial = initial
        self.final = final
    def at(self, t):
        f = min(float(t) / self.timesteps, 1.0)
        return self.initial + f*(self.final - self.initial)
