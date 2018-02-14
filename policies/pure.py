from torch.autograd import Variable

class PurePolicy(object):
    def __init__(self, model):
        self.model = model
    def get(self, state):
        self.model.eval()
        action_values = self.model(Variable(state))
        return action_values.max(1)[1].data[0]
