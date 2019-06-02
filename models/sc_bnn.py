import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import torch.nn.functional as F

class Sound_Classification_CNN(nn.Module):
	def __init__(self):
		super(Sound_Classification_CNN, self).__init__()

		self.conv1 = BinarizeConv2d(1, 32, kernel_size=3, stride=2)
		self.conv1_bn = torch.nn.BatchNorm2d(32)
		self.conv2 = BinarizeConv2d(32, 64, kernel_size=3, stride=2)
		self.conv2_bn = torch.nn.BatchNorm2d(64)
		#self.pool = nn.MaxPool2d(2,2)

		self.linear = BinarizeLinear(3840, 500, bias=True)
		self.out = BinarizeLinear(500, 10, bias=True)

	def forward(self,x):
		x = F.relu(self.conv1_bn(self.conv1(x)))
		x = F.relu(self.conv2_bn(self.conv2(x)))
		x = x.view(-1, 3840)
		x = F.relu(self.linear(x))
		y_pred = self.out(x)

		return y_pred

class Sound_Classification_MLP(nn.Module):
	def __init__(self):
		super(Sound_Classification_MLP,self).__init__()

		self.h = BinarizeLinear(1280, 500, bias=True)
		self.out = BinarizeLinear(500, 10, bias=True)
		self.bn = nn.BatchNorm1d(500)

	def forward(self,x):
		x = F.relu(self.bn(self.h(x)))
		y_pred = self.out(x)

		return y_pred