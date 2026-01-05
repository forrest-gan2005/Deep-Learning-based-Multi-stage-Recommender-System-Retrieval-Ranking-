import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    """
    Standard Pytorch Dataset for MovieLens 1M interactions
    """
    def __init__(self, df):
        # We take userId and movieId as input features
        # These are already LabelEncoded to 0, 1, 2, ... in preprocess.py
        self.features = df[['userId', 'movieId']].values.astype('int64')
        
        # Convert rating to binary label
        self.labels = (df['rating'].values >= 4).astype('float32')
        
    def __len__(self):
        return len(self.labels)
        
        
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])
        


