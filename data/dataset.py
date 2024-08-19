import torch
from torch.utils.data import Dataset

torch.manual_seed(42)

class Matrix(Dataset):
    def __init__(self, matrix_size, size):
        """
        Args:
            matrix_size (tuple): matrix size。
            size (int): dataset size
        """

        self.matrix_size = matrix_size
        self.size = size
        self.matrix = self.matrix_initialization()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        return self.matrix[idx]
    
    def matrix_initialization(self):
        matrix = torch.rand(self.size, *self.matrix_size)
        return matrix


if __name__ == '__main__':
    mymatrix = Matrix(matrix_size=(2,2), size=3)
    # print(mymatrix.matrix.shape)
    # print(len(mymatrix))

    for matrix in mymatrix:
        print(matrix)



