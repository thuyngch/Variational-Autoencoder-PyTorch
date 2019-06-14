#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#  VAEGT
#------------------------------------------------------------------------------
class VAEGT(nn.Module):
	def __init__(self, in_dims=784, hid1_dims=100, hid2_dims=64, num_classes=10, negative_slope=0.1):
		super(VAEGT, self).__init__()
		self.in_dims = in_dims
		self.hid1_dims = hid1_dims
		self.hid2_dims = hid2_dims
		self.num_classes = num_classes
		self.negative_slope = negative_slope

		# Encoder
		self.encoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(in_dims, 512)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(512, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 128)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
		]))
		self.fc_mu = nn.Linear(128, hid1_dims)
		self.fc_var = nn.Linear(128, hid1_dims)

		# Conditioner
		self.conditioner = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(num_classes, 16)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(16, 32)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(32, hid2_dims)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
		]))

		# Decoder
		self.decoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(hid1_dims+hid2_dims, 128)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(128, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 512)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer4', nn.Linear(512, in_dims)),
			('sigmoid', nn.Sigmoid()),
		]))

		self._init_weights()

	def forward(self, x, y):
		if self.training:
			# Encode input
			h = self.encoder(x)
			mu, logvar = self.fc_mu(h), self.fc_var(h)
			hx = self._reparameterize(mu, logvar)
			# Encode label
			y_onehot = self._onehot(y)
			hy = self.conditioner(y_onehot)
			# Hidden representation
			h = torch.cat([hx, hy], dim=1)
			# Decode
			y = self.decoder(h)
			return y, mu, logvar
		else:
			hx = self._represent(x)
			hy = self.conditioner(self._onehot(y))
			h = torch.cat([hx, hy], dim=1)
			y = self.decoder(h)
			return y

	def generate(self, y):
		hy = self.conditioner(self._onehot(y))
		hx = self._sample(y.shape[0]).type_as(hy)
		h = torch.cat([hx, hy], dim=1)
		y = self.decoder(h)
		return y

	def _represent(self, x):
		h = self.encoder(x)
		mu, logvar = self.fc_mu(h), self.fc_var(h)
		hx = self._reparameterize(mu, logvar)
		return hx

	def _reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size()).type_as(mu)
		z = mu + std * esp
		return z

	def _onehot(self, y):
		y_onehot = torch.FloatTensor(y.shape[0], self.num_classes)
		y_onehot.zero_()
		y_onehot.scatter_(1, y, 1)
		return y_onehot

	def _sample(self, num_samples):
		return torch.FloatTensor(num_samples, self.hid1_dims).normal_()

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
	model = VAEGT()
	model.eval()

	input = torch.rand([1, 784])
	label = torch.tensor([[1]])

	output = model(input, label)
	print(output.shape)
