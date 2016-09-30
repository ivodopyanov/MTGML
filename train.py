from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model, Graph
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU

import numpy as np
import pandas as pd
import sys
import time

draftType = "KLD_KLD_KLD"

pack_card_count = 249
picked_card_count = 249
#BFZ_BFZ_BFZ = 249
#OGW_OGW_BFZ = 432
#ORI_ORI_ORI = 252
#GPT_GPT_GPT = 165
#RAV_RAV_RAV = 286
#SOI_SOI_SOI = 282
#EMA_EMA_EMA = 249
#EMN_EMN_SOI = 487
#KLD_KLD_KLD = 249

pack_size = 15
picked_size = 45

batch_size = 256


def convertPack(pack):
	res = np.zeros((pack.shape[0],pack_card_count))
	for i in range(pack.shape[0]):
		for j in range(pack_size):
			if pack[i,j]!=0:
				pos = pack[i,j]-1
				res[i,pos] = 1
	return res

def convertPicked(picked):
	res = np.zeros((picked.shape[0],picked_card_count))
	for i in range(picked.shape[0]):
		for j in range(picked_size):
			if picked[i,j]!=0:
				pos = picked[i,j]-1
				res[i,pos] += 1
	return res

def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
	pos = y[i]
        Y[i, pos] = 1.
    return Y


model = Graph()
model.add_input(name='pack', input_shape=(pack_card_count,))
model.add_input(name='picked', input_shape=(picked_card_count,))
model.add_node(Dropout(0.2), name='pickedDropout', input='picked')

model.add_node(Dense(1024), name='picked1', input='pickedDropout')
model.add_node(PReLU(), name='picked1Act', input='picked1')
model.add_node(BatchNormalization(), name='picked1Norm', input='picked1Act')
model.add_node(Dropout(0.5), name='picked1Dropout', input='picked1Norm')

model.add_node(Dense(1024), name='picked2', input='picked1Dropout')
model.add_node(PReLU(), name='picked2Act', input='picked2')
model.add_node(BatchNormalization(), name='picked2Norm', input='picked2Act')
model.add_node(Dropout(0.5), name='picked2Dropout', input='picked2Norm')

model.add_node(Dense(1024), name='picked3', input='picked2Dropout')
model.add_node(PReLU(), name='picked3Act', input='picked3')
model.add_node(BatchNormalization(), name='picked3Norm', input='picked3Act')
model.add_node(Dropout(0.5), name='picked3Dropout', input='picked3Norm')

model.add_node(Dense(1024), name='picked4', input='picked3Dropout')
model.add_node(PReLU(), name='picked4Act', input='picked4')
model.add_node(BatchNormalization(), name='picked4Norm', input='picked4Act')
model.add_node(Dropout(0.5), name='picked4Dropout', input='picked4Norm')

#model.add_node(Dense(512), name='picked5', input='picked4Dropout')
#model.add_node(PReLU(), name='picked5Act', input='picked5')
#model.add_node(BatchNormalization(), name='picked5Norm', input='picked5Act')
#model.add_node(Dropout(0.5), name='picked5Dropout', input='picked5Norm')

#model.add_node(Dense(512), name='picked6', input='picked5Dropout')
#model.add_node(PReLU(), name='picked6Act', input='picked6')
#model.add_node(BatchNormalization(), name='picked6Norm', input='picked6Act')
#model.add_node(Dropout(0.5), name='picked6Dropout', input='picked6Norm')

model.add_node(Dense(pack_card_count), name='sum1', inputs=['picked4Dropout', 'pack'], merge_mode='concat')
model.add_node(Activation('softmax'), name='res', inputs=['sum1','pack'], merge_mode='mul')
model.add_output(name='output', input='res')
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(optimizer, {'output':'categorical_crossentropy'})


originalTrainData = pd.read_csv(draftType+"_data.csv", header=None)
earlyDraftsCutOff = 45*1500
trainData = originalTrainData.ix[earlyDraftsCutOff:,:]
trainData = trainData.iloc[np.random.permutation(len(trainData))]
packs = trainData.ix[:,:(pack_size-1)].values
picked = trainData.ix[:,pack_size:(pack_size+picked_size-1)].values
Y = trainData.ix[:,(pack_size+picked_size):].values

#Optional - throw away first drafts where players haven't learned new format yet
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
		packs_train = convertPack(packs[batch_start:batch_end])
		picked_train = convertPicked(picked[batch_start:batch_end,:])
		Y_train = to_categorical(Y[batch_start:batch_end,:], pack_card_count+1)[:,1:]
		trainError=model.train_on_batch(data = {'pack':packs_train, 'picked':picked_train, 'output':Y_train})[0]
		#trainError=model.train_on_batch([packs_train, picked_train], [Y_train])
		sumError+=trainError
		sys.stdout.write('\r'+'train batch '+str(batch)+'/'+str(batch_count-1)+' error '+"{0:.4f}".format(sumError/(batch+1)))
		sys.stdout.flush()
	sys.stdout.write('\n')

def test(model, packs, picked, Y, start, end, batch_size):
	sumError = 0
	batch_count=(trainData.shape[0]-end)/batch_size
	adjMatrix = np.zeros(shape=(picked_card_count, picked_card_count))
	for batch in range((trainData.shape[0]-end)/batch_size):
		batch_start = end+batch*batch_size
		batch_end = min(end+(batch+1)*batch_size, trainData.shape[0])
		packs_valid = convertPack(packs[batch_start:batch_end])
		picked_valid = convertPicked(picked[batch_start:batch_end,])
		Y_valid = to_categorical(Y[batch_start:batch_end,], pack_card_count+1)[:,1:]
		testError=model.test_on_batch(data = {'pack': packs_valid, 'picked': picked_valid, 'output':Y_valid})[0]
		#testError=model.test_on_batch([packs_valid, picked_valid], [Y_valid])
		sumError+=testError
		cl = model.predict(data={'pack': packs_valid, 'picked':picked_valid})['output']
		actual = np.argmax(cl, axis=1)
		expected = np.argmax(Y_valid, axis=1)
		for i in range(batch_end-batch_start):
			adjMatrix[expected[i],actual[i]]+=1
		sys.stdout.write('\r'+'val batch '+str(batch)+'/'+str(batch_count-1) + ' error '+"{0:.4f}".format(sumError/(batch+1)))
		sys.stdout.flush()
	sys.stdout.write('\nAccuracy: {0:.4f}'.format(np.trace(adjMatrix)/np.sum(adjMatrix)))
	sys.stdout.write('\n')




for epoch in range(100):
	sys.stdout.write('\nEpoch '+str(epoch+1)+'\n')
	sys.stdout.flush()
	start_time = time.time()
	train(model, packs, picked, Y, start, end, batch_size)
	test(model, packs, picked, Y ,start, end, 100)
	end_time = time.time()
	sys.stdout.write('Time: '+str(end_time-start_time)+'\n')
	sys.stdout.flush()




test(model, packs, picked, 0, start, 100)


modelName = draftType+'/'+draftType+'_arch_A1'
textfile = open(modelName+'.txt','w')
textfile.write(model.to_json())
textfile.close()
model.save_weights(modelName+'.h5', overwrite=True)
