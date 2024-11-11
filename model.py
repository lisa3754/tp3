import torch
import torch.nn as nn
import torch.optim as optim
import Vocab as Voczb

class Model(nn.Module):
    def __init__(self, input_size, output_size):    
        super().__init__()
        self.W = nn.Linear(input_size, 256)  #input_size = k * dim_embeddings
        self.relu = nn.ReLU()
        self.U = nn.Linear(256, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, k, dim_embeddings) -> (batch_size, k * dim_embeddings)
        x = self.W(x)
        x = self.relu(x)
        x = self.U(x)
        #x = self.softmax(x)
        return x
    
class ModelE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, k):
        super().__init__()
        
        self.E = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(k * embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.E(x)  #(batch_size, k, embedding_dim), récupère lignes des indices
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
