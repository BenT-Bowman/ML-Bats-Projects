import torch
import torch.nn as nn
import torch.nn.functional as F
class Countception(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(Countception, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Inception module 1
        self.inception1 = self._inception_module(64, 64, 64, 32, 32, 32)

        # Inception module 2
        # self.inception2 = self._inception_module(128, 128, 128, 64, 64, 64)

        # # Inception module 3
        # self.inception3 = self._inception_module(256, 256, 256, 128, 128, 128)

        # Counting layer
        self.counting_layer = nn.Conv2d(32, 1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1*40*40, 1)

    def _inception_module(self, num_filters1, num_filters2, num_filters3, num_filters4, num_filters5, num_filters6):
        inception_module = nn.Sequential(
            nn.Conv2d(self.input_shape[0], num_filters1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters2, num_filters3, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(num_filters3, num_filters4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_filters4, num_filters5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters5, num_filters6, kernel_size=5, padding=2),
            nn.ReLU()
        )
        return inception_module

    def forward(self, x):
        x = self.inception1(x)
        # print(x.shape)
        # x = self.inception2(x)
        # print(x.shape)
        # x = self.inception3(x)
        # print(x.shape)
        x = self.counting_layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    i = torch.randn(19, 3, 40, 40)
    # model = MoE(12)
    model = Countception(1, (3, 40, 40))
    print(model(i))