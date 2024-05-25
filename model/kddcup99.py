from torch import nn

class KDD(nn.Module):
    def __init__(self):
        super(KDD, self).__init__()
        num_inputs, num_hiddens, num_outputs = 40, 128, 23
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, 2 * num_hiddens),
            nn.ReLU(),
            nn.Linear(2 * num_hiddens, num_outputs)
        )

    def forward(self, x):
        x = self.model(x)
        return x