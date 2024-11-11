from model import ModelE
from Vocab import Vocab
import fonctions

emb_dim = 100
vocab = Vocab(corpus_filenam='Le_comte_de_Monte_Cristo.train.tok')

k = 10
input_dim = k*vocab.emb_dim #k*d = nombre de mots dans chaque séquence * dim_embeddings
vocab_size = vocab.vocab_size  # taille du vocabulaire en sortie
model = ModelE(vocab_size, embedding_dim=emb_dim, output_size=vocab_size, k=k)

# Entraîner le modèle avec un fichier de données et la classe Vocab
fonctions.train_modelE(model, dataset='Le_comte_de_Monte_Cristo.train.tok', vocab=vocab, num_epochs=10, batch_size=64, learning_rate=0.1, k=k)

# data = fonctions.text2list('Le_comte_de_Monte_Cristo.train.unk5.tok', vocab, input_dim)
# print(f"Nombre de séquences dans les données : {len(data)}")