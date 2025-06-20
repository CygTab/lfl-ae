from torch import nn


class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hid1, hid2, hid3, hid4, hid5):
        super(MultiOutputRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid1)
        self.layer2 = nn.Linear(hid1, hid2)
        self.layer3 = nn.Linear(hid2, hid3)
        self.layer4 = nn.Linear(hid3, hid4)
        self.layer5 = nn.Linear(hid4, hid5)
        self.output_layer = nn.Linear(hid5, output_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x