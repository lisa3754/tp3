import torch
import torch.nn.functional as F
import fonctions
from model import Model  
from Vocab import Vocab  


def generate_text(model, vocab, longueur, k):
    text = '<s>' #texte du début
    indices_text = [vocab.get_word_index(text) for i in range(k)] #dimension d'entrée

    for i in range(longueur+k-2):
        input_seq = indices_text[-k:] if len(indices_text) >= k else indices_text #garder seulement les derniers k mots pour la prédiction

        X = torch.tensor(input_seq).float()

        with torch.no_grad():
            output = model(X)
            probabilities = F.softmax(output, dim=0) #distribution probabilité mots suivants

        index_mot_suivant = torch.multinomial(probabilities, 1).item() #tire mot aléatoirement
        #index_mot_suivant = torch.argmax(probabilities)
        indices_text.append(index_mot_suivant)

        if vocab.get_word(index_mot_suivant) == '</s>': #stop si token fin de phrase
            break
    indices_text=indices_text[k-2:] #on garde <s> <s> du début
    generated_text = ' '.join([vocab.get_word(idx) for idx in indices_text])
    return generated_text

if __name__ == "__main__":
    model_path = 'optimized_model.pth'
    vocab = Vocab(emb_filename='embeddings-word2vecofficial.train.unk5.txt')
    input_dim = 10
    output_dim = vocab.vocab_size
    model = fonctions.load_model(model_path, input_dim, output_dim)

    generated_text = generate_text(model, vocab, longueur=50, k=input_dim) #génère text de longueur 50 ou moins
    print("Texte généré :")
    print(generated_text)
    #print(output_dim)