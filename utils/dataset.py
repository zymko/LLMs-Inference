import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import os

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        return self.queries[idx], self.keys[idx], self.values[idx]
    
    def matrix_initialization(self, seed, matrix_size):
        torch.manual_seed(seed)
        matrix = torch.rand(self.size, *matrix_size)
        return matrix

class GeneratedMatrix(Dataset):
    def __init__(self, query_dir, key_dir, value_dir):
        """
        Initialize the dataset with directories containing query, key, and value matrices.
        """
        self.query_dir = query_dir
        self.key_dir = key_dir
        self.value_dir = value_dir
        
        # List all files in the directories (assuming all directories have the same number of files)
        self.query_files = sorted(os.listdir(query_dir))
        self.key_files = sorted(os.listdir(key_dir))
        self.value_files = sorted(os.listdir(value_dir))
        
        # Verify that all directories have the same number of files
        assert len(self.query_files) == len(self.key_files) == len(self.value_files), "Mismatch in number of files in directories."
    
    def __len__(self):
        """
        Return the total number of samples (slices).
        """
        return len(self.query_files)
    
    def __getitem__(self, idx):
        """
        Return the query, key, and value matrices for the given index.
        """
        # Load the corresponding query, key, and value matrices from the .npy files
        query = np.load(os.path.join(self.query_dir, self.query_files[idx]))
        key = np.load(os.path.join(self.key_dir, self.key_files[idx]))
        value = np.load(os.path.join(self.value_dir, self.value_files[idx]))
        
        # Convert numpy arrays to torch tensors
        query = torch.tensor(query, dtype=torch.float32)
        key = torch.tensor(key, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.float32)
        
        return query, key, value


if __name__ == '__main__':
    # Paths to the directories containing the matrices
    query_dir = 'dataset/query'
    key_dir = 'dataset/key'
    value_dir = 'dataset/value'
    
    dataset = GeneratedMatrix(query_dir, key_dir, value_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for i, (query, key, value) in enumerate(dataloader):
        print(f"Sample {i}: Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
        if i >= 2:  
            break







