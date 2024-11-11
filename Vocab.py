import torch
import numpy as np
import torch
import sys

class Vocab:
    def __init__(self, **kwargs):

        self.dico_voca = {} #map les mots à leur indice correspondant
        self.word_array = [] #stock mots dans l'ordre auquel ils sont ajoutés
        if "emb_filename" in kwargs : #si le fichier embedding est fourni
            with open(kwargs["emb_filename"],'r') as fi:
                ligne = fi.readline() #lecture de la ligne courante -> ici la première
                ligne = ligne.strip() #supprime espaces..
                (self.vocab_size, self.emb_dim) = map(int,ligne.split(" "))
                self.matrice = torch.zeros((self.vocab_size, self.emb_dim))
                indice_mot = 0
        
                ligne = fi.readline()
                ligne = ligne.strip()
                while ligne != '': #tant que le fichier n'est pas complètement lu
                    splitted_ligne = ligne.split()
                    self.dico_voca[splitted_ligne[0]] = indice_mot
                    self.word_array.append(splitted_ligne[0])
                    for i in range(1,len(splitted_ligne)):
                        self.matrice[indice_mot, i-1] = float(splitted_ligne[i]) #embedding du mot numéro indice mot
                    indice_mot += 1
                    ligne = fi.readline() #ligne suivante est lu (à partir de la position actuelle du curseur)
                    ligne = ligne.strip()
        else: #si pas fourni, on construit le voc avec le corpus donné
            fichier_corpus = kwargs["corpus_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 0
            with open(fichier_corpus,'r') as fi: #gère automatiquement fermeture fichier une fois bloc with terminé : besoin d'appeler explicitement fi.close()
                for line in fi:
                    line = line.rstrip()
                    tokens = line.split(" ")
                    for token in tokens:
                        if token not in self.dico_voca :
                            self.word_array.append(token)
                            self.dico_voca[token] = nb_tokens
                            nb_tokens += 1
            self.vocab_size = nb_tokens
            print("vocab size =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim)) #matrice embeddings initialisée à zéro

    def get_word_index(self, mot):
        if not mot in self.dico_voca:
            return None
        return self.dico_voca[mot]
                
    def get_word_index2(self, mot):
        if not mot in self.dico_voca:
            return self.dico_voca['<unk>']
        return self.dico_voca[mot]
                
    def get_emb(self, mot):
        if not mot in self.dico_voca:
            return None
        return  self.matrice[self.dico_voca[mot]]
    
    def get_emb_torch(self, indice_mot):
        # OPTIMISATION: no verificaiton allows to get embeddings a bit faster
        #if indice_mot < 0 or indice_mot >= self.matrice.shape()[0]: # not valid index
        #    return None
        #return self.matrice[indice_mot]
        return self.matrice[indice_mot]
        
    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect

    def get_word(self, index):
        if index < len(self.word_array):
            return self.word_array[index]
        else:
            return None
    

#le chat est sur le toit
#le chien est dans la maison

#self.dico_voca = {'le': 0, 'chat': 1, 'est': 2, 'sur': 3, 'toit': 4, 'chien': 5, 'dans': 6, 'la': 7, 'maison': 8}
#self.word_array = ['le', 'chat', 'est', 'sur', 'toit', 'chien', 'dans', 'la', 'maison']

#self.matrice sera une matrice de zéros de taille (vocab_size, emb_dim), ici (9, 100) si emb_dim=100.