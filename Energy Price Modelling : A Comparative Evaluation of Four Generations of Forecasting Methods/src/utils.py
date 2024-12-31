import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict
from itertools import combinations
import pickle
from torch import optim
import torch.nn as nn
import os
import torch
from adabelief_pytorch import AdaBelief
import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import geopandas as gpd


#Helper Functions related to model training: ################################################################
def _acquire_device(args):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device

def _select_optimizer(model,args,current_model):
        if current_model!='Basisformer':
            model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
             para1 = [param for name,param in model.named_parameters() if 'map_MLP' in name]
             para2 = [param for name,param in model.named_parameters() if 'map_MLP' not in name]
             # optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
             model_optim = AdaBelief([{'params':para1,'lr':5e-3},{'params':para2,'lr':args.learning_rate}], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False) 
            
        return model_optim

def _select_criterion():
        criterion = nn.MSELoss()
        return criterion

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

def create_args(current_model,df):
    #Parser FEDFORMER: ########################################################################################
    if current_model=='FEDformer':
        parser = argparse.ArgumentParser(description='FEDformer family for Time Series Forecasting')

        # basic config
        parser.add_argument('--is_training', type=int, default=1, help='status')
        parser.add_argument('--task_id', type=str, default='test', help='task id')
        parser.add_argument('--model', type=str, default='FEDformer',
                            help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

        # supplementary config for FEDformer model
        parser.add_argument('--version', type=str, default='Fourier',
                            help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
        parser.add_argument('--mode_select', type=str, default='random',
                            help='for FEDformer, there are two mode selection method, options: [random, low]')
        parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
        parser.add_argument('--L', type=int, default=3, help='ignore level')
        parser.add_argument('--base', type=str, default='legendre', help='mwt base')
        parser.add_argument('--cross_activation', type=str, default='tanh',
                            help='mwt cross atention activation function tanh or softmax')

        # data loader
        parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                                    'S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                                    'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        # parser.add_argument('--cross_activation', type=str, default='tanh'

        # Define input and output sizes of model: depending on current data:
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2,
                            help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2,
                            help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2,
                            help='output size')



        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=3, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

        args = parser.parse_args()
        #True if torch.cuda.is_available() and args.use_gpu else False
        #if args.use_gpu and args.use_multi_gpu:
        #    args.dvices = args.devices.replace(' ', '')
        #    device_ids = args.devices.split(',')
        #    args.device_ids = [int(id_) for id_ in device_ids]
        #    args.gpu = args.device_ids[0]
        #DONE With Parser Fedformer! ################################################## 

    elif current_model=='Autoformer':
        parser = argparse.ArgumentParser(description='Autoformer')

        # basic config
        parser.add_argument('--is_training', type=int, default=1, help='status')
        parser.add_argument('--model_id', type=str,  default='test', help='model id')
        parser.add_argument('--model', type=str,  default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

        # model define
        parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
        parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')

        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')

        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        args = parser.parse_args()
    
    elif current_model=='TimeMixer':
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int, default=1, help='status')
        parser.add_argument('--model_id', type=str,default='test', help='model id')
        parser.add_argument('--model', type=str, default='Autoformer',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str,  default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # model define
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')
        parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default='avg',
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                            help='whether to use future_temporal_feature; True 1 False 0')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        parser.add_argument('--comment', type=str, default='none', help='com')

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        args = parser.parse_args()
    
    elif current_model=='Informer':
        parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

        parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

        parser.add_argument('--data', type=str, default='ETTh1', help='data')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
        parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
        parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')
        
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
        parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
        parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test',help='exp description')
        parser.add_argument('--loss', type=str, default='mse',help='loss function')
        parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

        #parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        #parser.add_argument('--gpu', type=int, default=0, help='gpu')
        #parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        #parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

        args = parser.parse_args()

    elif current_model=='TSMixer':
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int,  default=1, help='status')
        parser.add_argument('--model_id', type=str, default='test', help='model id')
        parser.add_argument('--model', type=str,  default='Autoformer',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str,  default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        #parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=48,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False, 
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
        args = parser.parse_args()
    
    elif current_model=='DLinear' or current_model=='NLinear':
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

        # basic config
        parser.add_argument('--is_training', type=int,  default=1, help='status')
        parser.add_argument('--train_only', type=bool,  default=False, help='perform training on full input dataset without validation and testing')
        parser.add_argument('--model_id', type=str,  default='test', help='model id')
        parser.add_argument('--model', type=str, default=current_model,
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


        # DLinear
        parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')

        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        args = parser.parse_args()
    elif current_model=='PatchTST':
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

        # random seed
        parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

        # basic config
        parser.add_argument('--is_training', type=int,  default=1, help='status')
        parser.add_argument('--model_id', type=str,  default='test', help='model id')
        parser.add_argument('--model', type=str,  default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str,  default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


        # DLinear
        #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')

        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        args = parser.parse_args()

    elif current_model=='Basisformer':
        parser = argparse.ArgumentParser(description='Time series prediction - Basisformer')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        parser.add_argument('--is_training', type=bool, default=True, help='train or test')
        parser.add_argument('--device', type=int, default=0, help='gpu dvice')

        # data loader
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--data', type=str, default='custom', help='dataset type')
        parser.add_argument('--root_path', type=str, default='all_six_datasets/traffic', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                                'S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                            'b:business days, w:weekly, m:mondfthly], you can also use more detailed freq like 15min or 3h')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=96, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        # parser.add_argument('--cross_activation', type=str default='tanh'

        # model define
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--heads', type=int, default=16, help='head in attention')
        parser.add_argument('--d_model', type=int, default=100, help='dimension of model')
        parser.add_argument('--N', type=int, default=10, help='number of learnable basis')
        parser.add_argument('--block_nums', type=int, default=2, help='number of blocks')
        parser.add_argument('--bottleneck', type=int, default=2, help='reduction of bottleneck')
        parser.add_argument('--map_bottleneck', type=int, default=20, help='reduction of mapping bottleneck')

        # optimization
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
        parser.add_argument('--tau', type=float, default=0.07, help='temperature of infonce loss')
        parser.add_argument('--loss_weight_prediction', type=float, default=1.0, help='weight of prediction loss')
        parser.add_argument('--loss_weight_infonce', type=float, default=1.0, help='weight of infonce loss')
        parser.add_argument('--loss_weight_smooth', type=float, default=1.0, help='weight of smooth loss')


        #checkpoint_path
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


        #parser.add_argument('--check_point',type=str,default='checkpoint',help='check point path, relative path')

        args = parser.parse_args()

    elif current_model=='Crossformer':
        parser = argparse.ArgumentParser(description='CrossFormer')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        parser.add_argument('--data', type=str,  default='ETTh1', help='data')
        parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  
        parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

        parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
        parser.add_argument('--out_len', type=int, default=96, help='output MTS length (\tau)')
        parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
        parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
        parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

        parser.add_argument('--data_dim', type=int, default=df.shape[1]-2, help='Number of dimensions of the MTS data (D)')
        parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
        parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
        parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

        parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
        parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')

        parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

        args = parser.parse_args()

    elif current_model=='Quatformer':

        parser = argparse.ArgumentParser(description='Quatformer for Time Series Forecasting')

        # basic config
        parser.add_argument('--is_training', type=int, default=1, help='status')
        parser.add_argument('--model_id', type=str, default='test', help='model id')
        parser.add_argument('--model', type=str, default='Quatformer',
                            help='model name, options: [Quatformer, Quatformer_without_Decoup_Attn]')

        # data loader
        parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

        # model define
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')

        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--order', type=int, default=1, help='polynormial order in trend normalization')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

        # periods settings
        parser.add_argument('--period_type', type=str, default='invariant', 
                            help='period type, options: [invariant, variant]')
        parser.add_argument('--n_periods', type=int, default=2, help='num of periods')

        # regularization coefficients
        parser.add_argument('--lambda_1', type=float, default=0., help='regularization coefficient of omegas')
        parser.add_argument('--lambda_2', type=float, default=0., help='regularization coefficient of thetas')


        args = parser.parse_args()

    elif current_model=='TSL_model':
        parser = argparse.ArgumentParser(description='TimesNet')
        parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int,  default=1, help='status')
        parser.add_argument('--model_id', type=str,  default='test', help='model id')
        parser.add_argument('--model', type=str, default='Autoformer',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=df.shape[1]-2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=df.shape[1]-2, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=df.shape[1]-2, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=48,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False, 
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
        
        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

        args = parser.parse_args()
    else:
        parser='None'

    return args
#############################################################################################################

#Helper Functions related to results visualization: #########################################################
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

# Create a function to plot heatmaps using Cartopy
def plot_heatmap(metrics, metric_name, model_name,countries):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent to focus on Europe
    ax.set_extent([-25, 45, 34.5, 71], crs=ccrs.PlateCarree())

    # Add countries borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Normalize the metric values
    metric_values = np.array([metrics[country][metric_name] for country in metrics])
    norm = plt.Normalize(metric_values.min(), metric_values.max())

    # Create a colormap
    cmap = ListedColormap(['darkgreen', 'green', 'yellow', 'orange', 'red'])

    for country in countries.itertuples():
        country_name = country.name
        if country_name in metrics:
            value = metrics[country_name][metric_name]
            color = cmap(norm(value))

            # Plot the country with the corresponding color
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color, edgecolor='black')

            # Annotate the country with the metric value
            x, y = country.geometry.representative_point().x, country.geometry.representative_point().y
            ax.text(x, y, f'{int(value)}', ha='center', fontsize=8, transform=ccrs.PlateCarree())
        else:
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), 
                              facecolor="None", edgecolor='white',alpha=1.0)
           

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=50, ticks=np.linspace(metric_values.min(), metric_values.max(), 10))
    cbar.ax.set_ylabel(f'{metric_name}', rotation=270, labelpad=15)

    #plt.title(f'{model_name} Model ')
    plt.show()


# Create a function to plot heatmaps using individual scales for each model
def plot_heatmap_individual_scale(smape_data, model_name,countries,model_generations):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent to focus on Europe
    ax.set_extent([-25, 45, 34.5, 71], crs=ccrs.PlateCarree())

    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Normalize the SMAPE values for this specific model
    smape_values = np.array([smape_data[country] for country in smape_data if not np.isnan(smape_data[country])])
    norm = plt.Normalize(smape_values.min(), smape_values.max())

    # Create a colormap
    cmap = ListedColormap(['darkgreen', 'green', 'yellow', 'orange', 'red'])

    for country in countries.itertuples():
        country_name = country.name
        if country_name in smape_data:
            value = smape_data[country_name]
            if not np.isnan(value):
                color = cmap(norm(value))

                # Plot the country with the corresponding color
                ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color, edgecolor='black')

                # Annotate the country with the SMAPE value (formatted to two decimal places)
                x, y = country.geometry.representative_point().x, country.geometry.representative_point().y
                ax.text(x, y, f'{value:.2f}', ha='center', fontsize=8, transform=ccrs.PlateCarree())

    # Adjust plot layout to make space for colorbar
    fig.subplots_adjust(right=0.85)

    # Add colorbar with individual scale for each model
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Adjust position and size of the colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('SMAPE', rotation=270, labelpad=15)

    # Set title centered above the map
    generation = model_generations.get(model_name, '')
    ax.set_title(f'{generation}: {model_name} model', fontsize=16, loc='center', pad=20)

    plt.show()


def plot_average_smape_heatmap(smape_data,countries):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent to focus on Europe
    ax.set_extent([-25, 45, 34.5, 71], crs=ccrs.PlateCarree())

    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Normalize the SMAPE values for the average data
    smape_values = np.array([smape_data[country] for country in smape_data if not np.isnan(smape_data[country])])
    norm = plt.Normalize(smape_values.min(), smape_values.max())

    # Create a colormap
    cmap = ListedColormap(['darkgreen', 'green', 'yellow', 'orange', 'red'])

    for country in countries.itertuples():
        country_name = country.name
        if country_name in smape_data:
            value = smape_data[country_name]
            if not np.isnan(value):
                color = cmap(norm(value))

                # Plot the country with the corresponding color
                ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color, edgecolor='black')

                # Annotate the country with the average SMAPE value (formatted to two decimal places)
                x, y = country.geometry.representative_point().x, country.geometry.representative_point().y
                ax.text(x, y, f'{value:.2f}', ha='center', fontsize=8, transform=ccrs.PlateCarree())

    # Adjust plot layout to make space for colorbar
    fig.subplots_adjust(right=0.85)

    # Add colorbar with the correct size
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Adjust position and size of the colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('Average SMAPE', rotation=270, labelpad=15)

    # Set title centered above the map
    ax.set_title('Average SMAPE heatmap across all model generations', fontsize=16, loc='center', pad=20)

    plt.show()


def visualize_smape_geomaps(timeSeries_benchmark_results):

    geojson_url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'
    countries = gpd.read_file(geojson_url)

    # List of European countries and Russia (since the dataset doesn't have 'continent')
    european_countries = [
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium',
        'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece',
        'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia',
        'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco',
        'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal',
        'Romania', 'Russia', 'San Marino', 'Republic of Serbia', 'Slovakia', 'Slovenia', 'Spain',
        'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'
    ]

    # Filter the countries GeoDataFrame for European countries and Russia
    countries = countries[countries['name'].isin(european_countries)]
    models = ['DLinear', 'NLinear', 'TSMixer', 'Autoformer', 'Basisformer', 'Informer', 'PatchTST', 'Quatformer', 'Chronos', 'TimesFM', 'ARIMA']
    smape_results = timeSeries_benchmark_results.copy()

    # Initialize a dictionary to store SMAPE values for each country and model
    smape_results['EU_Country'] = smape_results['EU_Country'].replace({
        'Czechia': 'Czech Republic',
        'Serbia': 'Republic of Serbia'
    })

    country_model_smape = {}

    # Loop over each model
    for model in models:
        # Filter the DataFrame for the current model
        model_data = smape_results[smape_results['Time_Series_Model'] == model]

        # Initialize an inner dictionary to store SMAPE values by country for the current model
        country_model_smape[model] = {}

        # Loop over each country in european_countries
        for country in european_countries:
            # Filter data for the specific country
            country_data = model_data[model_data['EU_Country'] == country]

            # Check if there's any data for this country
            if not country_data.empty:
                # Assuming 'Achieved_SMAPE' is the column for SMAPE values
                smape_value = country_data['Achieved_SMAPE'].mean()  # Get the mean SMAPE value if multiple rows exist
                country_model_smape[model][country] = smape_value
            else:
                # If no data, set to NaN
                country_model_smape[model][country] = np.nan


    # List of models you want to plot
    selected_models = ['PatchTST', 'TSMixer', 'TimesFM', 'ARIMA']

    # Dictionary mapping models to their generations
    model_generations = {
        'ARIMA': '1st generation',
        'TSMixer': '2nd generation',
        'PatchTST': '3rd generation',
        'TimesFM': '4th generation'
    }

    # Initialize a dictionary to store SMAPE values for each country and model
    smape_results['EU_Country'] = smape_results['EU_Country'].replace({
        'Czechia': 'Czech Republic',
        'Serbia': 'Republic of Serbia'
    })

    # Compute the average SMAPE across all models for each country
    average_smape_per_country = {}

    # Initialize a dictionary to store SMAPE values for each country and model
    smape_results['EU_Country'] = smape_results['EU_Country'].replace({
        'Czechia': 'Czech Republic',
        'Serbia': 'Republic of Serbia'
    })

    # Loop through each country and compute the average SMAPE across all models
    for country in european_countries:
        smape_values = []

        # Collect SMAPE values from all models for this country
        for model in models:
            if country in country_model_smape[model] and not np.isnan(country_model_smape[model][country]):
                smape_values.append(country_model_smape[model][country])

        # Calculate the average SMAPE for this country (if there are valid SMAPE values)
        if smape_values:
            average_smape_per_country[country] = np.mean(smape_values)
        else:
            # If no SMAPE values exist for this country, assign NaN
            average_smape_per_country[country] = np.nan


    # Plot the heatmap for average SMAPE across all models
    plot_average_smape_heatmap(average_smape_per_country,countries)

    # Generate heatmaps for the selected models using individual scales
    for model_name in selected_models:
        plot_heatmap_individual_scale(country_model_smape[model_name], model_name,countries,model_generations)
#############################################################################################################


#Helper function related to results evaluation: #############################################################
def friedman_aligned_ranks_test(data: np.ndarray) -> Dict[str, float]:
    # Original implementation
    row_means = np.mean(data, axis=1, keepdims=True)
    centered_data = data - row_means
    ranks = np.array([stats.rankdata(row) for row in centered_data])
    rank_sums = np.sum(ranks, axis=0)
    n, k = data.shape
    test_statistic = (12 / (n * k * (k + 1))) * np.sum(rank_sums**2) - 3 * n * (k + 1)
    df = k - 1
    p_value = stats.chi2.sf(test_statistic, df)
    # return test_statistic, p_value
    return {"statistic": test_statistic, "p_value": p_value, "df": df}

def post_hoc_procedures(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """
    Perform multiple post hoc procedures for a pairwise comparison.
    
    Parameters:
    data : np.ndarray
        A 2D array where each row represents a country and each column represents a model's metrics.
    alpha : float
        Significance level.
    
    Returns:
    Dict[str, Dict[str, float]]
        A dictionary containing the results of each post hoc procedure.
    """
    n, k = data.shape
    assert k == 2, "Post hoc procedures are designed for pairwise comparison only"
    
    # Calculate aligned observations and ranks
    grand_mean = np.mean(data)
    aligned_data = data - np.mean(data, axis=1)[:, np.newaxis] + grand_mean
    ranks = stats.rankdata(aligned_data.flatten()).reshape(n, k)
    
    # Calculate R (sum of ranks for each treatment)
    R = np.sum(ranks, axis=0)
    
    # Calculate the difference
    diff = abs(R[0] - R[1])
    
    # Calculate the standard error
    SE = np.sqrt(n * k * (k + 1) / 6)
    
    # Degrees of freedom
    df = (k - 1) * (n - 1)
    
    # Holland procedure
    holland_cv = stats.t.ppf(1 - alpha / 2, df) * SE
    holland_sig = diff > holland_cv
    
    # Rom procedure
    rom_cv = stats.t.ppf(1 - alpha / (2 * k), df) * SE
    rom_sig = diff > rom_cv
    
    # Finner procedure
    finner_cv = stats.norm.ppf(1 - alpha / 2) * SE
    finner_sig = diff > finner_cv
    
    # Li procedure
    li_cv = np.sqrt(2 * stats.f.ppf(1 - alpha, 1, df)) * SE
    li_sig = diff > li_cv
    
    return {
        "Holland": {"Critical Value": holland_cv, "Is Significant": holland_sig},
        "Rom": {"Critical Value": rom_cv, "Is Significant": rom_sig},
        "Finner": {"Critical Value": finner_cv, "Is Significant": finner_sig},
        "Li": {"Critical Value": li_cv, "Is Significant": li_sig}
    }

def pairwise_benchmark_vs_patchtst_friedman_aligned(metrics_data: Dict[str, List[float]], 
                                                    models: List[str], 
                                                    countries: List[str]) -> pd.DataFrame:
    """
    Perform pairwise Friedman aligned rank tests comparing PatchTST against each other model,
    followed by multiple post hoc procedures.
    
    Parameters:
    smape_data : Dict[str, List[float]]
        A dictionary where keys are model names and values are lists of metrics values for each country.
    models : List[str]
        List of model names, including PatchTST.
    countries : List[str]
        List of country names.
    
    Returns:
    pd.DataFrame
        A DataFrame containing the pairwise test results and summary statistics.
    """
    results = []
    patchtst_metrics = np.array(metrics_data["PatchTST"])
    
    for model in models:
        if model == "PatchTST":
            continue
        
        model_metrics = np.array(metrics_data[model])
        pairwise_data = np.column_stack((patchtst_metrics, model_metrics))
        
        friedman_result = friedman_aligned_ranks_test(pairwise_data)
        post_hoc_results = post_hoc_procedures(pairwise_data)
        
        # Calculate aligned ranks for this pair
        grand_mean = np.mean(pairwise_data)
        aligned_data = pairwise_data - np.mean(pairwise_data, axis=1)[:, np.newaxis] + grand_mean
        ranks = stats.rankdata(aligned_data.flatten()).reshape(pairwise_data.shape)
        aligned_ranks = np.sum(ranks, axis=0)
        
        mean_metrics_patchtst = np.mean(patchtst_metrics)
        mean_metrics_model = np.mean(model_metrics)
        
        result = {
            "Compared Model": model,
            "PatchTST Mean metrics": mean_metrics_patchtst,
            f"{model} Mean metrics": mean_metrics_model,
            "PatchTST Aligned Rank": aligned_ranks[0],
            f"{model} Aligned Rank": aligned_ranks[1],
            "Friedman Aligned Test Statistic": friedman_result["statistic"],
            "Friedman Aligned P-value": friedman_result["p_value"]
        }
        
        for proc, proc_result in post_hoc_results.items():
            result[f"{proc} Critical Value"] = proc_result["Critical Value"]
            result[f"{proc} Is Significant"] = proc_result["Is Significant"]
        
        results.append(result)
    
    return pd.DataFrame(results)
#############################################################################################################