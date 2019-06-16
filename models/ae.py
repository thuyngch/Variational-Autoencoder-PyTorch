#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#  AE
#------------------------------------------------------------------------------
class AE(nn.Module):
	def __init__(self, in_dims=784, hid_dims=100, negative_slope=0.1):
		super(AE, self).__init__()
		# Encoder
		self.encoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(in_dims, 512)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(512, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 128)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer4', nn.Linear(128, hid_dims)),
			('relu4', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
		]))
		# Decoder
		self.decoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(hid_dims, 128)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(128, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 512)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer4', nn.Linear(512, in_dims)),
			('sigmoid', nn.Sigmoid()),
		]))
		self._init_weights()

	def forward(self, x):
		z = self.encoder(x)
		y = self.decoder(z)
		return y

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


#------------------------------------------------------------------------------
#   Test bench
#------------------------------------------------------------------------------
if __name__ == "__main__":
	model = AE(in_dims=784, hid_dims=100, negative_slope=0.1)
	model.eval()

	input = torch.rand([1, 784])
	output = model(input)
	print(output.shape)
