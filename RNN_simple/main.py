import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")


"""
Data preparation
"""
class TextDataset(Dataset):
    """
    Text Dataset
    Text Dataset Class
    
    This class is in charge of managing text data as vectors
    Data is saved as vectors (not as text)
    Attributes
    ----------
    seq_length - int: Sequence length
    chars - list(str): List of characters
    char_to_idx - dict: dictionary from character to index
    idx_to_char - dict: dictionary from index to character
    vocab_size - int: Vocabulary size
    data_size - int: total length of the text
    """
    def __init__(self, text_data: str, seq_length: int = 25) -> None:
        """
        Inputs
        ------
        text_data: Full text data as string
        seq_length: sequence length. How many characters per index of the dataset.
        """
        self.chars = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)
        # useful way to fetch characters either by index or char
        self.idx_to_char = {i:ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch:i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(text_data)
    
    @property
    def X_string(self) -> str:
        """
        Returns X in string form
        """
        return self.vector_to_string(self.X)
        
    def __len__(self) -> int:
        """
        We remove the last sequence to avoid conflicts with Y being shifted to the left
        This causes our model to never see the last sequence of text
        which is not a huge deal, but its something to be aware of
        """
        return int(len(self.X) / self.seq_length -1)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X and Y have the same shape, but Y is shifted left 1 position
        """
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length

        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx+1:end_idx+1]).float()
        return X, y
    
    def string_to_vector(self, name: str) -> list[int]:
        """
        Converts a string into a 1D vector with values from char_to_idx dictionary
        Inputs
        name: Name as string
        Outputs
        name_tensor: name represented as list of integers (1D vector)
        sample:
        >>> string_to_vector('test')
        [20, 5, 19, 20]
        """
        vector = list()
        for s in name:
            vector.append(self.char_to_idx[s])
        return vector

    def vector_to_string(self, vector: list[int]) -> str:
        """
        Converts a 1D vector into a string with values from idx_to_char dictionary
        Inputs
        vector: 1D vector with values in the range of idx_to_char
        Outputs
        vector_string: Vector converted to string
        sample:
        >>> vector_to_string([20, 5, 19, 20])
        'test'
        """
        vector_string = ""
        for i in vector:
            vector_string += self.idx_to_char[i]
        return vector_string
    
    def tensor_to_string(self, vector: torch.Tensor) -> str:
        """
        Converts a 1D vector into a string with values from idx_to_char dictionary
        Inputs
        vector: 1D vector with values in the range of idx_to_char
        Outputs
        vector_string: Vector converted to string
        sample:
        >>> tensor_to_string([ 1., 37., 37., 33., 28.,  0.,  0.,  0.,  0.,  0.])
        'Apple     '
        """
        vector_string = ""
        for i in vector:
            vector_string += self.idx_to_char[int(i.item())]
        return vector_string

def padding_all_food_to_same_length(raw_data_lst: list[str]) -> tuple[int, list[str]]:
    """
    Returns a list of food names with the same length
    Inputs
    ------
    raw_data_lst: List of food names
    Outputs
    -------
    padded_data_lst: List of food names with the same length
    """
    max_len = max([len(i) for i in raw_data_lst]) + 1 # this is for shift left
    padded_data_lst = [i + " " * (max_len - len(i)) for i in raw_data_lst]
    return max_len, padded_data_lst



if __name__ == '__main__':
    raw_data_lst = ["Apple", "Burger", "Chicken", "Dumplings", "Egg", "Fries", "Grapes", "Honey", "Ice Cream", "Jelly", "Ketchup", "Lemon", "Milk", "Noodles", "Orange", "Pasta", "Salad", "Tomato", "Udon", "Vanilla", "Watermelon", "Xigua", "Yogurt", "Zucchini"]
    max_len, padded_data_lst = padding_all_food_to_same_length(raw_data_lst)
    text_data = "".join(padded_data_lst)

    seq_length = max_len - 1
    dataset = TextDataset(text_data, seq_length=seq_length)

    x, y = dataset[0]
    print(dataset.tensor_to_string(x))
    print(dataset.tensor_to_string(y))