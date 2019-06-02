import scipy.io as spio
import numpy as np
import os
import copy
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from skimage import io, transform

from models.sc_bnn import Sound_Classification_CNN, Sound_Classification_MLP

# Rescale Samples into fixed dimension
def rescale_sample(samples, h_rescale, w_rescale):
	num_sample, _ = samples.shape
	x = np.zeros((num_sample,h_rescale,w_rescale))
	
	# Rescale the sound feature representation to fix dimension,where the frequency dimension remains 
	# to 20 channels the temporal dimension changes to 64
	for i in range(num_sample):
		sample = samples[i,0]
		x[i] = transform.resize(sample, (h_rescale, w_rescale), mode='constant')
		
	return x

def train(train_x, train_labels, model, loss_fn, optimizer, num_epochs, device):
	num_sample = len(train_labels)
	train_x = Variable(torch.from_numpy(train_x), requires_grad=False)
	train_y = Variable(torch.from_numpy(train_labels), requires_grad=False) - 1

	train_x = train_x.view(num_sample,1,20,64) # CNN
	#train_x = train_x.view(num_sample,20*64) # MLP

	model.train()

	for iepoch in range(num_epochs):
		since = time.time()    

		# Training Loop
		inputs = train_x.type(torch.FloatTensor).to(device)
		train_y = train_y.type(torch.LongTensor).to(device) 

		# Forward Pass
		scores = model.forward(inputs)
		_, pred = torch.max(scores.data, dim=1)
		loss = loss_fn(scores, train_y)

		optimizer.zero_grad()
		loss.backward()  

		# load the stored full-precision weights for update
		for p in list(model.parameters()):
			if hasattr(p,'org'):
				p.data.copy_(p.org)

		optimizer.step()

		# the newly updated weight param will be stored in the p.org
		for p in list(model.parameters()):
			if hasattr(p,'org'):
				p.org.copy_(p.data.clamp_(-1,1))
		
		epoch_loss = loss.item()
		train_acc = torch.sum(pred == train_y).item()/float(num_sample)

		time_elapsed = time.time() - since

		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(iepoch+1, time_elapsed // 60, time_elapsed % 60))
		print('Loss: {:4f}, Train Accuracy: {:2f}'.format(epoch_loss, train_acc*100))

	return model

def test(test_x, test_labels, model, device):
	model.eval()

	num_sample = len(test_labels)

	test_x = Variable(torch.from_numpy(test_x), requires_grad=False)
	test_y = Variable(torch.from_numpy(test_labels), requires_grad=False) -1

	test_x = test_x.view(num_sample,1,20,64) # CNN
	#test_x = test_x.view(num_sample,20*64) # MLP

	inputs = test_x.type(torch.FloatTensor).to(device)
	test_y = test_y.type(torch.LongTensor).to(device) 

	# Forward Pass
	scores = model.forward(inputs)
	_, pred = torch.max(scores.data, dim=1)
	test_acc = torch.sum(pred == test_y).item()/float(num_sample)

	return test_acc

if __name__ == '__main__':
	# Load Data and Data Pre-processing
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"

	mat = spio.loadmat('data/pySoundClean.mat')
	FBETrainList = mat['FBETrainList']
	FBETestList = mat['FBETestList']
	train_labels = np.squeeze(mat['train_labels'])
	test_labels = np.squeeze(mat['test_labels'])

	h_rescale = 20
	w_rescale = 64
	num_sample = 200
	num_class = 10

	# Rescale the train and test spectrogram to uniform dimensionality
	train_x = rescale_sample(FBETrainList,h_rescale,w_rescale)
	test_x = rescale_sample(FBETestList,h_rescale,w_rescale)

	# CUDA configuration 
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	torch.multiprocessing.set_start_method("spawn")

	num_epochs = 100

	if device == 'cuda':
		cudnn.benchmark = True

	loss_fn = torch.nn.CrossEntropyLoss()

	# Models and training configuration 
	model = Sound_Classification_CNN()
	#model = Sound_Classification_MLP()
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0)
	
	# Model Training
	model = train(train_x, train_labels, model, loss_fn, optimizer, num_epochs, device)

	# Save the model
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')

	state = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
	}
	torch.save(state, 'checkpoint/sc_binary_cnn.pt')

	# Model Evaluation
	test_acc = test(test_x, test_labels, model, device)

	print('Test Accuracy is {:.2f}'.format(test_acc*100))