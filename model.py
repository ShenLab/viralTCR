import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from sklearn.metrics import pairwise_distances as pds
from keras.layers import concatenate 
from keras import optimizers
from keras.backend import slice
from keras.layers import Lambda
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors, datasets
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from Bio.Seq import Seq

data = pd.read_csv('VDJdb_TCR.tsv', sep='\t')
data['epi'] = data['Meta'].str.split('"epitope.id": "').str[1].str.split('"').str[0]
data = data.dropna()
#data = data[data["Species"] == 'HomoSapiens']
#data = data[data["Reference"] == 'PMID:28636592']
data_re = pd.DataFrame(columns = ['Species','TRA_CDR3','TRA_V','TRA_J', 'TRB_CDR3','TRB_V','TRB_J','MHC A','Epitope','Epitope gene','epi','Score'],index = list(set(data['complex.id'])))

for index, row in data.iterrows():
    sys.stdout.write('%d\r' %index)
    sys.stdout.flush()
    if row['Gene'] == 'TRA':
        data_re.loc[row['complex.id'], 'TRA_CDR3'] = row['CDR3']
        data_re.loc[row['complex.id'], 'TRA_V'] = row['V']
        data_re.loc[row['complex.id'], 'TRA_J'] = row['J']
    else:
        data_re.loc[row['complex.id'], 'TRB_CDR3'] = row['CDR3']
        data_re.loc[row['complex.id'], 'TRB_V'] = row['V']
        data_re.loc[row['complex.id'], 'TRB_J'] = row['J']
    data_re.loc[row['complex.id'], 'MHC A'] = row['MHC A']
    data_re.loc[row['complex.id'], 'Epitope'] = row['Epitope']
    data_re.loc[row['complex.id'], 'Epitope gene'] = row['Epitope gene']
    data_re.loc[row['complex.id'], 'epi'] = row['epi']
    data_re.loc[row['complex.id'], 'Species'] = row['Species']
    data_re.loc[row['complex.id'], 'Score'] = row['Score']
    
data_re = data_re.dropna()

model_dir = Path('uniref50_v2')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=-1)

f = open('tracdr.txt','r').readlines()
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

f = open('trbcdr.txt','r').readlines()
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

epi = []
ac3 = []
bc3 = []

for ind, out in data_re.iterrows():
	epi.append(out[8])
	ac3.append(out[1])
	bc3.append(out[4])

embed_ac1 = embed(ac1)
embed_ac2 = embed(ac2)
embed_ac3 = embed(ac3)

embed_bc1 = embed(bc1)
embed_bc2 = embed(bc2)
embed_bc3 = embed(bc3)

embed_epi = embed(epi)


sel = list(set(indac1) & set(indbc1))
sel.sort() 
train_data = []
for i,x in enumerate(sel):
	train_data.append([embed_ac1[ac1[i]], embed_ac2[ac2[i]], embed_ac3[ac3[i]], embed_bc1[bc1[i]], embed_bc2[bc2[i]], embed_bc3[bc3[i]], embed_epi[epi[i]]]) 
	sys.stdout.write('%d\r' %i)
	sys.stdout.flush() 

timesteps = 7
n_features = 1024
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape = (timesteps,n_features), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences = False))
model.add(RepeatVector(timesteps))
model.add(LSTM(64, activation='relu', return_sequences = True))
model.add(LSTM(256, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_data, epochs=100, verbose=1)
modelx = Model(inputs=model.inputs, outputs=model.layers[1].output)  
yhat = modelx.predict(train_data)

epg = list(data_re['Epitope gene']) 
epg2 = [epg[i] for i in sel] 
standard_embedding = umap.UMAP(n_neighbors=20,min_dist=0.0, n_components=2,random_state=42).fit_transform(yhat) 
fig, ax = plt.subplots() 
for i,g in enumerate(np.unique(epg2)):
	i = np.where(np.array(epg2) == g)
	if len(i) >50:
		ax.scatter(standard_embedding[i,0], standard_embedding[i, 1], label=g, s=8,alpha=0.5)
	else:
		ax.scatter(standard_embedding[i,0], standard_embedding[i, 1], s=8,alpha=0.5) 

ax.legend(fontsize=14) 
ax.tick_params(axis="x", labelsize=20) 
ax.tick_params(axis="y", labelsize=20) 
ax.set_xlabel("UMAP_1",fontsize = 24) 
ax.set_ylabel("UMAP_2",fontsize = 24) 
ax.grid(True) 
plt.savefig('umap_cluster.jpeg',dpi=300) 


rest = [np.where(epg != np.unique(epg2)[i])[0] for i in range(len(np.unique(epg2)))] 
keep = [np.where(epg == np.unique(epg2)[i])[0] for i in range(len(np.unique(epg2)))] 
train_neg = np.copy(train_data)
for i in range(len(keep)): 
	sys.stdout.write('%d\r' %i)
	sys.stdout.flush()
	neg_sel = np.array(random.sample(list(rest[i])*2,len(keep[i])))
	train_neg[keep[i],6] = train_data[neg_sel,6] 

x_train = np.concatenate((train_data,train_neg))  
x = modelx.predict(x_train) 
y = np.array([1]*len(train_data) + [0] * len(train_data)).reshape(-1,1) 

model2 = Sequential()
model2.add(Dense(32, input_dim=64, activation='relu'))
model2.add(Dense(16,  activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])                                                                        
model2.fit(x, y, batch_size=500, epochs=200)

modelx_json = modelx.to_json()                                                                                                        
with open("LSTM_Audoencoder.json", "w") as json_file:
	json_file.write(modelx_json)
	modelx.save_weights("LSTM_Audoencoder.h5")   

model2_json = model2.to_json()                                                                                                        
with open("DNN.json", "w") as json_file:
	json_file.write(model2_json)
	model2.save_weights("DNN.h5")                                                                                                      



