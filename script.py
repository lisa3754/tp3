corpusFileName = 'Le_comte_de_Monte_Cristo.train.tok'
outputFileName = 'Le_Comte_De_Monte_Cristo.train.unk5.tok'
threshold = 5

#fréquences des unigrams
unigram = {}
with open(corpusFileName, "r") as fi:
    for line in fi:
        tokens = line.split()
        for token in tokens:
            if token not in unigram:
                unigram[token] = 1
            else:
                unigram[token] += 1

with open(corpusFileName, "r") as fi, open(outputFileName, "w") as fo:
    for line in fi:
        tokens = line.split()
        new_tokens = [
            token if unigram[token] >= threshold else "<unk>"
            for token in tokens
        ]
        fo.write(" ".join(new_tokens) + "\n")

print(f"fichier sauvegardé : {outputFileName}")
