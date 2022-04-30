from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import fnmatch
import random
import numpy as np
class ChessPositionsDataset(Dataset):
    def __init__(self, csv_dir, score_path, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_dir = csv_dir
        self.score_path = score_path
        self.transform = transform
        self.len = 0
        buffer = np.empty((500000, 9, 8, 8))
        for i in range(0, 500000):
            buffer[i,:,:,:] =  self.load_position(i)
        self.data = buffer
        evals = []
        with open('data/stockfish_modified.TXT') as f:
            file_contents = f.readlines()
            for i in range(0, 500000):
                evals.append(file_contents[i])
        self.evals = evals
    def __len__(self):
        # if self.len == 0:
        #     self.len = len(fnmatch.filter(os.listdir(self.csv_dir), '*.*'))
        return 500000 - 1
    def __getitem__(self, index):
        board_pos = torch.from_numpy(self.data[index])
        # with open('data/stockfish_modified.TXT') as f:
        # with open(self.score_path) as f:
        # file_contents = f.readlines()
        try:
            eval =  torch.tensor(float(self.evals[index]))
            return (board_pos, eval)
        except ValueError:
            rand_index = random.randint(0, self.__len__())
            return self.__getitem__(rand_index)
    def load_position(self, idx):
        position_shape = (9, 8, 8)
        # retrieving data from file.
        # pos = np.loadtxt("data/board_positions/board"+str(idx)+'.csv')
        pos = np.loadtxt(os.path.join(self.csv_dir, "board"+str(idx)+'.csv'))
        # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
        # reshaping to get original matrice with original shape.
        loaded_pos = pos.reshape(pos.shape[0], pos.shape[1] // position_shape[2], position_shape[2])
        # print(loaded_pos)
        return loaded_pos
