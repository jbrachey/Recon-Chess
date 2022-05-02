import torch.nn as nn
import torch.nn.functional as F
class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x
class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=64, kernel_size=2, stride= (2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride= (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        # self.conv1 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1)
        # self.linear1 = nn.Linear(in_features=32 * 18, out_features=128),
        # self.linear2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, board_pos):
        # print(board_pos)
        return self.network(board_pos)
        # board_pos = self.conv1(board_pos)
        # # print(board_pos.shape)

        # board_pos = F.relu(board_pos)

        # board_pos = board_pos.reshape(-1, 32 * 18)
        # # print(board_pos.shape)
        # board_pos = self.linear1(board_pos)
        # board_pos = self.linear2(board_pos)
        return board_pos
