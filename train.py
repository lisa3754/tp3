from Vocab import Vocab
from model import Model
import fonctions

vocab = Vocab(emb_filename='embeddings-word2vecofficial.train.unk5.txt')

k = 10
input_dim = k*vocab.emb_dim #k*d = nombre de mots dans chaque séquence * dim_embeddings
output_dim = vocab.vocab_size  # taille du vocabulaire en sortie
model = Model(input_dim, output_dim)

# Entraîner le modèle avec un fichier de données et la classe Vocab
fonctions.train_model(model, dataset='Le_comte_de_Monte_Cristo.train.tok', vocab=vocab, num_epochs=10, batch_size=64, learning_rate=0.1, k=k)

# data = fonctions.text2list('Le_comte_de_Monte_Cristo.train.unk5.tok', vocab, input_dim)
# print(f"Nombre de séquences dans les données : {len(data)}")