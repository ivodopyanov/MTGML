from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.models import Graph
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2

import numpy as np
import pandas as pd
import sys
import time

draftType = "ORI_ORI_ORI"

pack_card_count = 282
picked_card_count = 282
#BFZ_BFZ_BFZ = 249
#OGW_OGW_BFZ = 432
#ORI_ORI_ORI = 252
#GPT_GPT_GPT = 165
#RAV_RAV_RAV = 286
#SOI_SOI_SOI = 282

pack_size = 15
picked_size = 45

batch_size = 256


def convertPicked(picked):
	res = np.zeros((picked.shape[0],picked_card_count))
	for i in range(picked.shape[0]):
		for j in range(picked_size):
			if picked[i,j]!=0:
				res[i,picked[i,j]-1] += 1
	return res

def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y



model = Graph()
model.add_input(name='pack', input_shape=(pack_card_count,))
model.add_input(name='picked', input_shape=(picked_card_count,))
model.add_node(Dropout(0.2), name='pickedDropout', input='picked')

model.add_node(Dense(512), name='picked1', input='pickedDropout')
model.add_node(PReLU(), name='picked1Act', input='picked1')
model.add_node(BatchNormalization(), name='picked1Norm', input='picked1Act')
model.add_node(Dropout(0.5), name='picked1Dropout', input='picked1Norm')

model.add_node(Dense(512), name='picked2', input='picked1Dropout')
model.add_node(PReLU(), name='picked2Act', input='picked2')
model.add_node(BatchNormalization(), name='picked2Norm', input='picked2Act')
model.add_node(Dropout(0.5), name='picked2Dropout', input='picked2Norm')

#model.add_node(Dense(1000), name='picked3', input='picked2Dropout')
#model.add_node(PReLU(), name='picked3Act', input='picked3')
#model.add_node(BatchNormalization(), name='picked3Norm', input='picked3Act')
#model.add_node(Dropout(0.5), name='picked3Dropout', input='picked3Norm')

model.add_node(Dense(512), name='picked4', input='picked2Dropout')
model.add_node(PReLU(), name='picked4Act', input='picked4')
model.add_node(BatchNormalization(), name='picked4Norm', input='picked4Act')
model.add_node(Dropout(0.5), name='picked4Dropout', input='picked4Norm')


model.add_node(Dense(pack_card_count), name='sum1', input='picked4Dropout')
model.add_node(Activation('softmax'), name='res', inputs=['sum1','pack'], merge_mode='mul')
model.add_output(name='output', input='res')
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(optimizer, {'output':'categorical_crossentropy'})


trainData = pd.read_csv(draftType+"/data.csv", header=None)
#Throw away first drafts where players haven't learned new format yet
earlyDraftsCutOff = 45*500
trainData = trainData.ix[earlyDraftsCutOff:,:]
trainData = trainData.iloc[np.random.permutation(len(trainData))]
packs = trainData.ix[:,:(pack_size-1)].values
picked = trainData.ix[:,pack_size:(pack_size+picked_size-1)].values
Y = trainData.ix[:,(pack_size+picked_size):].values


#And use latest drafts for validation
start = int(packs.shape[0]*0.0)
end=int(packs.shape[0]*0.9)





def train(model, packs, picked, Y, start, end, batch_size):
	sumError = 0
	batch_count=(end-start)/batch_size
	for batch in range(batch_count):
		batch_start = start+batch*batch_size
		batch_end = min(start+(batch+1)*batch_size, end)
		#Format card index in categorical one-hot data (for Y) or their sum (for picked and train)
		packs_train = to_categorical(packs[batch_start:batch_end],pack_card_count+1)[:,1:]
		picked_train = convertPicked(picked[batch_start:batch_end,:])
		Y_train = to_categorical(Y[batch_start:batch_end,:], pack_card_count+1)[:,1:]
		trainError=model.train_on_batch(data = {'pack':packs_train, 'picked':picked_train, 'output':Y_train})
		sumError+=trainError[0]
		sys.stdout.write('\r'+'train batch '+str(batch)+'/'+str(batch_count-1)+' error '+"{0:.4f}".format(sumError/(batch+1)))
		sys.stdout.flush()
	sys.stdout.write('\n')

def test(model, packs, picked, Y, start, end, batch_size):
	sumError = 0
	batch_count=(trainData.shape[0]-end)/batch_size
	for batch in range((trainData.shape[0]-end)/batch_size):
		batch_start = end+batch*batch_size
		batch_end = min(end+(batch+1)*batch_size, trainData.shape[0])
		packs_valid = to_categorical(packs[batch_start:batch_end,],pack_card_count+1)[:,1:]
		picked_valid = convertPicked(picked[batch_start:batch_end,])
		Y_valid = to_categorical(Y[batch_start:batch_end,], pack_card_count+1)[:,1:]
		testError=model.test_on_batch(data = {'pack': packs_valid, 'picked': picked_valid, 'output':Y_valid})
		sumError+=testError[0]
		sys.stdout.write('\r'+'val batch '+str(batch)+'/'+str(batch_count-1) + ' error '+"{0:.4f}".format(sumError/(batch+1)))
		sys.stdout.flush()
	sys.stdout.write('\n')



for epoch in range(50):
	sys.stdout.write('\nEpoch '+str(epoch+1)+'\n')
	sys.stdout.flush()
	start_time = time.time()
	train(model, packs, picked, Y, start, end, batch_size)
	test(model, packs, picked, Y ,start, end, 100)
	end_time = time.time()
	sys.stdout.write('Time: '+str(end_time-start_time)+'\n')
	sys.stdout.flush()



modelName = draftType+'/'+draftType+'_arch_AW'
textfile = open(modelName+'.txt','w')
textfile.write(model.to_json())
textfile.close()
model.save_weights(modelName+'.h5', overwrite=True)
