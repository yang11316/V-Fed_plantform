from torch import nn
import torch
class SECOM(nn.Module):
    def __init__(self):
        super(SECOM, self).__init__()
        num_inputs, num_hiddens, num_outputs = 590, 1024, 2
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens,bias=True),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8, num_outputs,bias=True)
        )

    def forward(self, x):
        x = torch.sigmoid(self.model(x))
        return x