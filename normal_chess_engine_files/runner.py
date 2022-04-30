from logging import root
from random import shuffle
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ChessPositionsDataset import ChessPositionsDataset
from ChessCNN import ChessCNN
# from Model import ChessCNN
import torch
import os
from torch.utils.data import DataLoader, Dataset

class Trainer:
    def __init__(
        self, 
        data_dir,
        model,
        device,
        batch_size: int = 250,
        lr: float = 0.0015,
        lr_decay: float = 0.001
    ):  
        self.data_dir = data_dir
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr = lr)
        self.device = device
        self.batch_size = batch_size

    def train(self, epochs):
        train_pos_dataset = ChessPositionsDataset(csv_dir = "data/board_positions", score_path = "data/stockfish_modified.TXT")
        train_set, val_set = torch.utils.data.random_split(train_pos_dataset, [400000, 100000])
        train_loader = DataLoader(train_set, batch_size = self.batch_size, shuffle = True, num_workers=2)
        test_loader = DataLoader(val_set, batch_size = self.batch_size, shuffle = False, num_workers=2)
        for epoch in range(epochs): 
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                board_pos, true_eval = data

                new_shape = (true_eval.size(0), 1)
                true_eval = true_eval.view(new_shape)

                board_pos = board_pos.to(device = self.device, dtype=torch.float)
                true_eval = true_eval.to(device = self.device, dtype=torch.float)
                # zero param grad
                self.optimizer.zero_grad()

                # forward + backprop + optimize
                # print(board_pos.size())
                guessed_eval = self.model(board_pos)
                # print(guessed_eval.size())
                loss = self.criterion(guessed_eval, true_eval)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
            with torch.no_grad():
                combined_accuracy = 0
                total = 0
                try:
                    for positions, evals in train_loader:
                        positions = positions.to(device = self.device, dtype=torch.float)
                        evals = evals.to(device = self.device, dtype=torch.float)
                        outputs = self.model(positions)
                        total += evals.size(0)
                        accuracy = torch.divide(outputs - evals, torch.divide(outputs + evals, 2) )
                        combined_accuracy += accuracy.sum().item()
                    print('average Accuracy of the network on the {} train images on training: {} %'.format(50000, 100 * combined_accuracy / total))
                except ValueError: 
                    print("error doing accuracy! for training")
            with torch.no_grad():
                try:
                    for positions, evals in test_loader:
                        positions = positions.to(device = self.device, dtype=torch.float)
                        evals = evals.to(device = self.device, dtype=torch.float)
                        # calculate outputs by running images through the network
                        outputs = self.model(positions)
                        # the class with the highest energy is what we choose as prediction
                        total += evals.size(0)
                        accuracy = torch.divide(outputs - evals, torch.divide(outputs + evals, 2) )
                        combined_accuracy += accuracy.sum().item()
                    print(f'Accuracy of the network on the 10000 test images: {100 * combined_accuracy / total} %')
                except ValueError: 
                    print("error doing accuracy! for test")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                "model_params/model_params",
            )
        print("Training finished!")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            "model_params/model_params",
        )
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ChessCNN()
trainer = Trainer(data_dir = "data", model = model, device = device, batch_size = 32, lr= 0.001, lr_decay= 0.001)
trainer.train(30000)