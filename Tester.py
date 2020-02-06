import os
import torch
import torch.nn as nn
import torch.optim as optim

class Tester(nn.Module):
    def __init__(self, model, device):
        super(Trainer, self).__init__()

        self.model = model
        self.model.to(device)

    def forward(self, input, target):

        self.model.zero_grad()
        output = self.model(input)
        loss = self.criterion(output, target)

        loss.backward()
        self.optim.step()

        return loss

    def load(self, state):
        self.model.load_state_dict(state['weight'])

    def save(self, dir, it):
        state_name = os.path.join(dir, 'quantlstm{}.pkl'.format(it))

        state = {
            'weight': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'iterations': it,
        }

        torch.save(state, state_name)