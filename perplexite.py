import torch
import torch.nn.functional as F
import math
import fonctions
from model import Model  
from Vocab import Vocab  
from tqdm import tqdm #voir le temps restant

def perplexite(model, list):
    total_log_prob = 0.0
    total_words = 0

    with torch.no_grad(): 
        for sequence in tqdm(list, desc="Calcul de la perplexité", unit="seq"):
            X = torch.tensor(sequence[:-1]).unsqueeze(0).float()  # k mots contexte
            y = sequence[-1]  # mot cible

            # prédire probabilités de chaque mot
            output = model(X)
            log_prob = F.log_softmax(output, dim=1) #pas de softmax à la fin du modèle (pour crossentropy loss training) + log pour faire somme

            # ajouter probabilité mot cible
            word_log_prob = log_prob[0, y].item()
            total_log_prob += word_log_prob
            total_words += 1

    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob) #passer du log à perplexité
    return perplexity

if __name__ == "__main__":

    model_path = 'optimized_model.pth'
    text = 'Le_Comte_de_Monte_Cristo.train.unk5.tok'

    vocab = Vocab(emb_filename='embeddings-word2vecofficial.train.unk5.txt')
    input_dim = 10
    output_dim = vocab.vocab_size
    model = fonctions.load_model(model_path, input_dim, output_dim)

    list = fonctions.text2list(text, vocab, input_dim)

    perplexity = perplexite(model, list)
    print(f"Perplexité du texte donné : {perplexity:.4f}")

    #1502.2965
