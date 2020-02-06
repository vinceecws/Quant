import os
import torch
import numpy as np
import pandas
from sklearn import preprocessing
from torch.utils.data import Dataset

class OHLCV(Dataset):
    def __init__(self, directory, window, lag=1, returns=False, scaled=None, binary=True):
        self._directory = directory
        self._window = window
        self._lag = lag
        assert(self._lag <= self._window), 'Lag size must be <= window size'
        self._returns = returns
        self._scaled = scaled
        self._binary = binary

        data = pandas.read_csv(directory, usecols=['Open', 'High', 'Low', 'Close', 'Volume'])[['Open', 'High', 'Low', 'Close', 'Volume']] #Extract O,H,L,C,V columns, ordered as such
        data = data.drop(np.where(data['Volume'] == 0.0)[0]) #Drop days with 0 market activity

        self._data = data.values

        if self._scaled:
            if self._scaled == 'std': #Standardize with mean=0 and standard deviation=1
                self._scaler = preprocessing.StandardScaler()
                self._data = self._scaler.fit_transform(self._data)
            elif self._scaled == 'minmax':
                self._scaler = preprocessing.MinMaxScaler()
                self._data = self._scaler.fit_transform(self._data)
            else:
                raise ValueError('Invalid value for arg: scaled')


    def __len__(self):
        return len(self._data) - self._window + 1

    def __getitem__(self, idx):

        if(idx >= self.__len__() or idx < -self.__len__()):
            raise IndexError('index out of bounds')
        elif(idx < 0):
            idx = self.__len__() + idx
            print(idx)

        data = self._data[idx : idx+self._window : self._lag]

        index = np.expand_dims(np.arange(1, self._window + 1), 1)
        data = np.concatenate((index, data), axis=1)

        if self._returns:
            returns = np.expand_dims((data[:, 4] - data[:, 1]), 1)
            data = np.concatenate((data, returns), axis=1)

        if self._binary:
            if data[-1, 4] > data[-1, 1]: #close > open (price increase)
                target = 1
            else: #close <= open (price decrease or no change)
                target = 0
        else:
            target = np.expand_dims(data[-1, :], 1)

        return np.transpose(data[:-1, :],(1,0)), target

    @property
    def directory(self):
        return self._directory
    
    @property
    def window(self):
        return self._window
    
    @property
    def lag(self):
        return self._lag
    
    @property
    def returns(self):
        return self._returns
    
    @property
    def scaled(self):
        return self._scaled
    
    @property
    def binary(self):
        return self._binary
    
    @property
    def indexed(self):
        return self._indexed
    
    @property
    def scaler(self):
        return self._scaler
    
    @property
    def data(self):
        return self._data

