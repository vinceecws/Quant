import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.ohlcv import OHLCV
from tqdm import tqdm
from Trainer import Trainer
from quantitative_model import QuantLSTM
from quantitative_model import QuantCNN
from adabound import AdaBound

def main(args):
    resume = args.resume
    batch_size = args.batch_size
    num_epoch = args.epoch
    chart_data_dir = args.chart_data_dir
    weight_dir = args.weight_dir
    decay = args.decay
    lr = args.lr
    momentum = args.momentum
    window = args.window

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QuantLSTM(3, 1, window, blocks=2, batch_size=batch_size, dropout=0.3, logistic=False).to(device)
    #model = QuantCNN(5, 2, filters_per_chn=2048, blocks=4, dropout=0.3, batchnorm=True, relu=True).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = AdaBound(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()

    trainset = OHLCV(chart_data_dir, window, returns=True, scaled='std', binary=False)
    dataloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size,
                    shuffle=True, num_workers=4, drop_last=True)
    trainer = Trainer(model, optimizer, criterion, device, decay)

    if resume:
        assert os.path.isfile(resume), "{} is not a file.".format(resume)
        state = torch.load(resume)
        trainer.load(state)
        it = state["iterations"]
        print("Checkpoint is loaded at {} | Iterations: {}".format(resume, it))

    else:
        it = 0

    for e in range(1, num_epoch + 1):
        prog_bar = tqdm(dataloader, desc="Epoch {}".format(e))
        sum_loss = 0.0
        for i, data in enumerate(prog_bar):
            input = torch.index_select(data[0], 1, torch.tensor([1,4,6])).permute(0, 2, 1).float().to(device) #extract volume and returns column
            target = torch.squeeze(torch.index_select(data[1], 1, torch.tensor([4])).float().to(device), 2)

            loss, output = trainer(input, target)
            loss = loss.item() / batch_size

            sum_loss += loss
            prog_bar.set_postfix(Loss=loss, Avg_loss=sum_loss/(i+1))

            it += 1
            
            if it % 100 == 0:
                print('Target: {}'.format(target))
                print('Output: {}'.format(output))
            if it % 2000 == 0:
                print('Saving checkpoint...')
                trainer.save(weight_dir, it)
                print("Checkpoint is saved at {} | Iterations: {}".format(weight_dir, it))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Load weight from file")
    parser.add_argument("--lr", type=float, default=1e-4, required=False, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.8, required=False, help="SGD Momentum")
    parser.add_argument("--decay", type=float, default=0.7, required=False, help="Learning rate decay")
    parser.add_argument("--batch_size", type=int, default=4, required=False, help="Batch size for SGD")
    parser.add_argument("--epoch", type=int, default=5, required=False, help="No. of epoch to train")
    parser.add_argument("--chart_data_dir", type=str, default="./chart_data/fx/EURUSD_Candlestick_1_D_BID_06.06.2009-01.06.2019.csv", required=False, help="File: Chart data in CSV")
    parser.add_argument("--weight_dir", type=str, default="./weights", required=False, help="Directory: Weights")
    parser.add_argument("--window", type=int, default=61, required=False, help="Window size for timeseries")
    args = parser.parse_args()

    main(args)