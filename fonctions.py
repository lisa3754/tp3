import Vocab
import torch
from model import Model
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import random

#dict = Vocab(emb_filename="corpus.txt")
def text2list(path_file, vocab, k):
    sequences = []  
    with open(path_file, 'r') as file:
        text = file.read()  # lit contenu fichier
    words = text.split()  # séparer texte en mots
    indices = [vocab.get_word_index(word) for word in words if vocab.get_word_index(word) is not None]
    for i in range(len(indices) - k):
        sequence = indices[i:i + k + 1]  #k+1 mots
        sequences.append(sequence)
    return sequences

def train_model(model, dataset, vocab, num_epochs=10, batch_size=32, learning_rate=0.01, k=3):
   
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data = text2list(dataset, vocab, k)
    
    for epoch in range(num_epochs):
        random.shuffle(data) #mélange les batch
        
        epoch_loss = 0.0
        num_batches = len(data) // batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            X = torch.tensor([x[:-1] for x in batch])  # Séquences de k mots : batch*k
            y = torch.tensor([x[-1] for x in batch])   # Mot cible
            
            X_embeddings = torch.stack([torch.stack([vocab.get_emb_torch(indice) for indice in seq]) for seq in X]) #batch*k*d

            optimizer.zero_grad()
            output = model(X_embeddings.float())

            #one_hot_labels = torch.stack([vocab.get_one_hot(vocab.get_word(indice)) for indice in y])

            # Calculer la perte
            #loss = criterion(output, one_hot_labels)
            loss = criterion(output, y)
            
            # Backward pass et optimisation
            loss.backward()  # Calcul des gradients
            optimizer.step()  # Mise à jour des paramètres
            
            epoch_loss += loss.item()

        # Afficher la perte moyenne pour cette epoch
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Sauvegarder les paramètres optimisés
    torch.save(model.state_dict(), 'optimized_model.pth')
    print("Modèle sauvegardé sous 'optimized_model.pth'")

def train_modelE(model, dataset, vocab, num_epochs=10, batch_size=32, learning_rate=0.01, k=3):
   
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data = text2list(dataset, vocab, k)
    
    for epoch in range(num_epochs):
        random.shuffle(data) #mélange les batch
        
        epoch_loss = 0.0
        num_batches = len(data) // batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            X = torch.tensor([x[:-1] for x in batch])  # Séquences de k mots : batch*k
            y = torch.tensor([x[-1] for x in batch])   # Mot cible
            
            optimizer.zero_grad()
            output = model(X.float())

            #one_hot_labels = torch.stack([vocab.get_one_hot(vocab.get_word(indice)) for indice in y])

            # Calculer la perte
            #loss = criterion(output, one_hot_labels)
            loss = criterion(output, y)
            
            # Backward pass et optimisation
            loss.backward()  # Calcul des gradients
            optimizer.step()  # Mise à jour des paramètres
            
            epoch_loss += loss.item()

        # Afficher la perte moyenne pour cette epoch
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Sauvegarder les paramètres optimisés
    torch.save(model.state_dict(), 'optimized_model.pth')
    print("Modèle sauvegardé sous 'optimized_model.pth'")

def load_model(model_path, input_size, output_size):
    model = Model(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  #mode évaluation
    return model


