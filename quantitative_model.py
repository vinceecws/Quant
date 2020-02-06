import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TaskDecoder(nn.Module):
	def __init__(self, binary=False, batchnorm=True, relu=True):
		self.binary = binary
		self.batchnorm = batchnorm
		self.relu = relu
		self.Conv1 = nn.Conv2d(1, 1, kernel_size=3)

		if self.batchnorm:
			self.BN1 = nn.BatchNorm2d(1)

		if self.relu:
			self.relu = nn.ReLU()

		self.Conv2 = nn.Conv2d(1, 1, kernel_size=3)
		self.AvgPool = nn.AdaptiveAvgPool1d(1)

	def forward(self, x):
		x = self.En(x)
		x = self.Conv1(x)

		if self.batchnorm:
			x = self.BN1(x)

		if self.relu:
			x = self.relu(x)

		x = self.Conv2(x)
		x = self.AvgPool(x)

		if self.binary:
			return F.softmax(x)
		else:
			return x


class QuantModel(nn.Module):
	def __init__(self, freeze=False, batchnorm=True, pretrained=False):
		self.freeze = freeze
		self.batchnorm = batchnorm
		self.pretrained = pretrained
		self.En = models.resnet34(pretrained=pretrained)
		self.En = nn.Sequential(*list(self.En.children())[:-2])
		self.De1 = TaskDecoder(binary=True, batchnorm=self.batchnorm, relu=True) #price up/down
		self.De2 = TaskDecoder(binary=False, batchnorm=self.batchnorm, relu=True) #volatility


	def forward(self, x):
		x = self.En(x)
		out1 = self.De1(x)
		out2 = self.De2(x)

		return (out1, out2)


class QuantMultiLoss(nn.Module):
	def __init__(self, alpha=0.5, beta=0.5):
		self.alpha = alpha
		self.beta = beta
		self.crossentropy = nn.CrossEntropyLoss()
		self.mse = nn.MSELoss()

	def forward(self, x, y_classification, y_regression):
		classification_loss = self.alpha * self.crossentropy(x, y_classification)
		regression_loss = self.beta * self.mse(x, y_regression)

		return (torch.sum(classification_loss, regression_loss), classification_loss, regression_loss)

class QuantLSTM(nn.Module):
	def __init__(self, in_chn, out_chn, window, blocks=2, batch_size=4, batch_first=True, dropout=0.3, batchnorm=True, relu=True, logistic=True):
		super(QuantLSTM, self).__init__()
		self.in_chn = in_chn #input_chn
		self.out_chn = out_chn #output_chn
		self.window = window #window_size for timeseries
		self.batch_size = batch_size
		self.dropout = dropout
		self.batchnorm = batchnorm
		self.relu = relu
		self.logistic = logistic

		layers = []
		for i in range(blocks):
			layers.append(QuantLSTMBlock(self.in_chn, self.in_chn, batch_first=batch_first, dropout=self.dropout, batchnorm=self.batchnorm, relu=self.relu))
		self.LSTMlayers = nn.Sequential(*layers)

		self.linear = nn.Linear(self.in_chn * (self.window - 1), self.out_chn)

		if self.batchnorm:
			self.BN = nn.BatchNorm1d(self.out_chn)

		if self.relu:
			self.ReLU = nn.LeakyReLU()

		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, states=None):
		#pass sequence of L x N x input_chn, (hidden_states, cell_states) -> L x N x output_chn, (new_hidden_states, new_cell_states)
		for block in self.LSTMlayers:
			x, states = block(x, states)

		x = self.linear(x.contiguous().view(self.batch_size, -1))

		if self.relu:
			x = F.relu(x)
		if self.batchnorm:
			x = self.BN(x)

		if self.logistic:
			x = F.softmax(x, dim=1)

		return x

class QuantLSTMBlock(nn.Module):
	def __init__(self, in_chn, out_chn, batch_first=True, dropout=0.3, batchnorm=True, relu=True):
		super(QuantLSTMBlock, self).__init__()
		self.in_chn = in_chn
		self.out_chn = out_chn
		self.batchnorm = batchnorm
		self.relu = relu

		self.LSTM = nn.LSTM(self.in_chn, self.out_chn, batch_first=batch_first)

		if self.batchnorm:
			self.BN = nn.BatchNorm1d(self.out_chn)

		if self.relu:
			self.ReLU = nn.LeakyReLU()

		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, states=None):
		x, states = self.LSTM(x, states)

		x = x.permute(0, 2, 1) #N * L * C -> N * C * L
		if self.batchnorm:
			x = self.BN(x)

		if self.relu:
			x = self.ReLU(x)

		x = self.dropout(x)
		x = x.permute(0, 2, 1) #N * C * L -> N * L * C

		return x, states

class QuantCNN(nn.Module):
	def __init__(self, in_chn, out_chn, filters_per_chn=32, blocks=2, dropout=0.3, batchnorm=True, relu=True):
		'''
		Note: Each feature/input channel will have it's own set of filters, since they represent datas in different domains
			  filters_per_chn = number of filters per feature/input channel
		Note: input length must be >= 4
		'''

		super(QuantCNN, self).__init__()
		self.in_chn = in_chn
		self.out_chn = out_chn
		self.blocks = blocks
		self.batchnorm = batchnorm
		self.relu = relu
		self.num_features = filters_per_chn*self.in_chn

		self.ConvIn = nn.Conv1d(self.in_chn, self.num_features, 3, stride=1, padding=0, dilation=1, groups=self.in_chn)
		if self.batchnorm:
			self.BN = nn.BatchNorm1d(self.num_features)

		if self.relu:
			self.ReLU = nn.LeakyReLU()

		layers = []
		for i in range(blocks):
			layers.append(QuantCNNBlock(self.num_features, dropout=dropout, batchnorm=self.batchnorm, relu=self.relu))

		self.ConvBlocks = nn.Sequential(*layers)

		self.dropout = nn.Dropout(p=dropout)
		self.AvgPool = nn.AdaptiveAvgPool1d(1)
		self.FC = nn.Linear(self.num_features, self.out_chn)

	def forward(self, x):
		x = self.ConvIn(x)
		if self.batchnorm:
			x = self.BN(x)

		if self.relu:
			x = self.ReLU(x)
			
		x = self.dropout(x)
		x = self.ConvBlocks(x)
		x = self.AvgPool(x)
		x = self.FC(x.view(-1, self.num_features))
		x = F.softmax(x, dim=1)

		return x

class QuantCNNBlock(nn.Module):
	def __init__(self, in_chn, dropout=0.3, batchnorm=True, relu=True):
		super(QuantCNNBlock, self).__init__()
		self.in_chn = in_chn
		self.batchnorm = batchnorm
		self.relu = relu

		#Retain separate filter for each channel of data
		self.Conv1 = nn.Conv1d(self.in_chn, self.in_chn, 3, stride=1, padding=0, dilation=1, groups=self.in_chn)
		self.ConvDown = nn.Conv1d(self.in_chn, self.in_chn, 2, stride=2, padding=0, dilation=1, groups=self.in_chn)

		if self.batchnorm:
			self.BN = nn.BatchNorm1d(self.in_chn)

		if self.relu:
			self.ReLU = nn.LeakyReLU()

		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.Conv1(x)
		if self.batchnorm:
			x = self.BN(x)

		if self.relu:
			x = self.ReLU(x)

		x = self.dropout(x)

		x = self.ConvDown(x)
		if self.batchnorm:
			x = self.BN(x)

		if self.relu:
			x = self.ReLU(x)

		x = self.dropout(x)

		return x



