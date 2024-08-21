import torch
from torch.utils.data import Dataset, DataLoader


class Matrix(Dataset):
    def __init__(self, matrix_sizes, size, seeds):
        """
        Args:
            matrix_size (tuple): matrix size。
            size (int): dataset size
            seeds (tuple): random seeds
        """

        self.matrix_size_q, self.matrix_size_k, self.matrix_size_v = matrix_sizes
        self.size = size
        self.seed_q, self.seed_k, self.seed_v = seeds
        self.queries = self.matrix_initialization(self.seed_q, self.matrix_size_q)
        self.keys = self.matrix_initialization(self.seed_k, self.matrix_size_k)
        self.values = self.matrix_initialization(self.seed_v, self.matrix_size_v)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        return self.queries[idx], self.keys[idx], self.values[idx]
    
    def matrix_initialization(self, seed, matrix_size):
        torch.manual_seed(seed)
        matrix = torch.rand(self.size, *matrix_size)
        return matrix


if __name__ == '__main__':
    mymatrix = Matrix(matrix_size=(2,2), size=100)
    dataloader = DataLoader(mymatrix, batch_size=32, shuffle=True, num_workers=4)
    for matrix in dataloader:
        print(matrix)




