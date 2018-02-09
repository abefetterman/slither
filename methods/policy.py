import numpy as np
from methods.scheduler import ExpoSchedule
from torch.autograd import Variable

class EpsPolicy(object):
    def __init__(self, model):
        self.schedule = ExpoSchedule(10000, 1.0, .1)
        self.model = model
    def get(self, state, t):
        eps = self.schedule.at(t)
        self.model.eval()
        action_values = self.model(Variable(state))
        if np.random.uniform(0,1) > eps:
            self.model.eval()
            # yes, this hot mess is how we get the argmax as an integer
            return action_values.max(1)[1].data[0]
        else:
            return np.random.randint(len(action_values[0]))
