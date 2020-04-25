import os
import sys
import time
import math
import keras
import random
import numpy as np
import pandas as pd
from sklearn import svm
from pathlib import Path
from Bio.Seq import Seq
from sklearn.decomposition import PCA
from scipy.spatial import distance
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.layers import concatenate 
from keras import optimizers
from keras.backend import slice
from keras.layers import Lambda
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors, datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import pairwise_distances as pds
from allennlp.commands.elmo import ElmoEmbedder
from keras.models import model_from_json
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from matplotlib_venn import venn3 

covpep = []
pepnam = []

def catchSeq(name, seq, k):
	gene = name.split('[gene=')[1].split(']')[0]
	kmers = {}
	for i in range(len(seq) - k + 1):
		kmer = seq[i:i+k]
		if kmer in kmers:
			kmers[kmer] += 1
		else:
			kmers[kmer] = 1
	sortedKmer = sorted(kmers.items(), reverse=True)
	for item in sortedKmer:
		covpep.append(item[0])
		pepnam.append(gene)

with open('virus_sequence.txt', 'r') as f:
	seq = ""
	key = ""
	for line in f.readlines():
		if line.startswith(">"):
			if key and seq:
				catchSeq(key, seq, 9)
			key = line[1:].strip()
			seq = ""
		else:
			seq += line.strip()
	catchSeq(key, seq, 9)

def abc(fa,fb):
    f = open(fa,'r').readlines()
    ac1 = []
    ac2 = []
    indac1 = []
    indac2 = []
    for i,x in enumerate(f[1:]):
        if i%3==0:
            ac1.append(x.split('\t')[1])
            indac1.append(int(x.split('\t')[0]))
        elif i%3==1:
            ac2.append(x.split('\t')[1])
            indac2.append(int(x.split('\t')[0]))
    f = open(fb,'r').readlines()
    bc1 = []
    bc2 = []
    indbc1 = []
    indbc2 = []
    for i,x in enumerate(f[1:]):
        if i%2==0:
            bc1.append(x.split('\t')[1])
            indbc1.append(int(x.split('\t')[0]))
        elif i%2==1:
            bc2.append(x.split('\t')[1])
            indbc2.append(int(x.split('\t')[0]))
    return ac1, ac2, bc1, bc2

test_tcr = pd.read_csv('TCR.csv')
test_ac1, test_ac2, test_bc1, test_bc2 = abc('test_tracdr.txt','test_trbcdr.txt')
test_ac3 = []
test_bc3 = []

for ind, out in test_tcr.iterrows():
    test_ac3.append(out[1])
    test_bc3.append(out[4])


model_dir = Path('uniref50_v2')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights, cuda_device = -1)

def s2v(seq):
    embed1 = seqvec.embed_sentence( list(seq) ) 
    protein_embd1 = torch.tensor(embed1).sum(dim=0).mean(dim=0) 
    return list(protein_embd1.detach().numpy())
def embed(l):
    value = []
    uni = list(set(l))
    for i, seq in enumerate(uni):
        sys.stdout.write('%d\r' %i)
        sys.stdout.flush()
        value.append(s2v(seq))
    return dict(zip(uni,value))

embed_ac1 = embed(test_ac1)
embed_ac2 = embed(test_ac2)
embed_ac3 = embed(test_ac3)

embed_bc1 = embed(test_bc1)
embed_bc2 = embed(test_bc2)
embed_bc3 = embed(test_bc3)

embed_epi = embed(covpep)

test_data = []
for i,x in enumerate(test_ac1):
    for j,y in enumerate(covpep):
        test_data.append([embed_ac1[test_ac1[i]], embed_ac2[test_ac2[i]], embed_ac3[test_ac3[i]], embed_bc1[test_bc1[i]], embed_bc2[test_bc2[i]], embed_bc3[test_bc3[i]], embed_epi[covpep[j]]])
        sys.stdout.write('%d\r' %i)
        sys.stdout.flush()

json_file = open('LSTM_Autoencoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelx = model_from_json(loaded_model_json)
modelx.load_weights("LSTM_Autoencoder.h5")

json_file = open('DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
model2.load_weights("DNN.h5")

def cov_test(t_covpep, t_embed_epi, sn):
    tdata = []
    print (sn)
    sel_index = list(test_tcr[test_tcr['Patient'] == sn].index)
    tac1 = [test_ac1[i] for i in sel_index]
    tac2 = [test_ac2[i] for i in sel_index]
    tac3 = [test_ac3[i] for i in sel_index]
    tbc1 = [test_bc1[i] for i in sel_index]
    tbc2 = [test_bc2[i] for i in sel_index]
    tbc3 = [test_bc3[i] for i in sel_index]
    tepi = t_covpep
    eepi = t_embed_epi
    #embed_epi[covpep[j]]
    for i,x in enumerate(tac1):
        for j,y in enumerate(tepi):
            tdata.append([embed_ac1[tac1[i]], embed_ac2[tac2[i]], embed_ac3[tac3[i]], embed_bc1[tbc1[i]], embed_bc2[tbc2[i]], embed_bc3[tbc3[i]],eepi[tepi[j]]])
            sys.stdout.write('%d\r' %i)
            sys.stdout.flush()

    tdata = np.array(tdata)
    print ('step 1')
    ytest = modelx.predict(tdata)
    print ('step 2')
    pred = model2.predict(ytest)
    return pred

sample = sorted(list(set(test_tcr['Patient'])))
output_cov = []
for sn in sample:
    output_cov.append(cov_test(covpep, embed_epi, sn))

sel_cov = [set([int(i%len(covpep)) for i in np.where(j>0.9)[0]]) for j in output_cov]


sel_con = sel_set[0]
sel_cri = []                                                                                                                                                              
for i in [1,3]:
	sel_cri +=  sel_set[i]
sel_oth = []                                                                                                                                                              
for i in [2,4,5,6,7,8,9,10]:
	sel_oth +=  sel_set[i]
	
fig, ax = plt.subplots(figsize=(6,3))
venn3([set(sel_con),set(sel_cri),set(sel_oth)],('Healthy(1)','Critical & Server(2)','Moderate(7)')) 
plt.savefig('venn3.jpeg',dpi=300) 


