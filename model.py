
#model creation
# import libraries
# !pip install pytorch-lightning
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from IPython.display import clear_output

import torch as tch
import torch
import torchvision as tv
from torchvision.transforms import transforms
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch import nn
from torch.nn import Sequential,Dropout, Linear, Identity, Conv2d, ReLU
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import SGD, RMSprop
from process_data import *
import os
torch.autograd.set_detect_anomaly(True)
#building the pytorch lighting class Model
class Custom_Model(pl.LightningModule):
    def __init__(self,model_name=None,every_b=3,best_acc=0.0,curr_ephochs=0,train_flag=False, train_dataroot=None, train_datapath=None, val_dataroot = None, val_datapath=None, test_dataroot=".", test_datapath="labels/test1.csv"):
        super(Custom_Model, self).__init__()
        self.train_dataroot,self.train_datapath, self.val_dataroot, self.val_datapath, self.test_dataroot, self.test_datapath, self.train_flag = train_dataroot, train_datapath, val_dataroot, val_datapath, test_dataroot, test_datapath, train_flag
        self.n_batch,self.n_v_batch,self.trn_lss, self.val_lss=0,0,0,0,
        self.model_name,self.every_b = model_name, every_b
        self.best_acc,self.epoch,self.still_epoch = best_acc,0,False
        self.total_epochs = curr_ephochs
        self.fc =  Sequential(OrderedDict([
                                          ('cassifier1', Linear(53081,500)),
                                        #   ('cassifier1', Linear(56937,100)),
                                          ('relu1', ReLU()),
                                        #   ('drop1', Dropout(0.25)),
                                          ('cassifier2', Linear(500,500)),
                                          ('relu2', ReLU()),
                                        #   ('drop2', Dropout(0.25)),
                                          ('classifier3', Linear(500, 500)),
                                          ('relu3', ReLU()),
                                        #   ('drop3', Dropout(0.1)),
                                          ('cassifier4', Linear(500,2)),
                                            ]))

    def forward(self, x):
        # print(f'try shit')
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        return  self.fc(x).view(x.shape[0],-1)

    def configure_optimizers(self):
        # print(f'init shit')
        return RMSprop([{'params' : self.parameters(), 'lr':0.000002}],
                       lr=0.000002, momentum=0.9)

    def my_loss(self, y_hat, y):
        # print('bit check')
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_size):
        # print(f'attempt run')
        if self.n_batch == 0 and not self.still_epoch:
        #   clear_output()
        # Clear the screen.
        #   os.system('clear')
          self.still_epoch = True
          print(f'<============= Epoch {self.epoch} ==============>')
          self.total_epochs += 1
        x,y = batch
        # print(x)
        #forward pass
        logits = self.forward(x)
        # print(f'logits -->  {logits}')
        #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y).cpu()
        del x, batch
        # print(f'y --> {y} pred --> {preds} loss --> {loss}')
        #add up train loss
        self.trn_lss += loss
        
        #save model state
        if self.model_name:
            model_info = {'state_dict_cnn' : self.state_dict(),'best_acc':self.best_acc,'epochs':self.total_epochs}
            torch.save(model_info, './models'+self.model_name+'_train.pt')
        
        #print out something if it is on every third batch
        if self.n_batch%self.every_b==(self.every_b-1):
            b_n = int(self.n_batch/self.every_b)+1
            of_ = self.dataset_sizes['train']//self.every_b
            print(f'batch {b_n}/{of_} loss {self.trn_lss/(b_n*self.every_b)}')
        self.n_batch+=1
        return {'loss' : loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def validation_step(self, batch, batch_size):
        # print(f'bit call')
        x,y = batch
        logits = self.forward(x)
        
        # #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y).cpu()
        # print(f'y --> {y} pred --> {preds} loss --> {loss}')
        #add up val loss
        self.val_lss += loss
        
        #print out something if it is on every n batch
        if self.n_v_batch%self.every_b==(self.every_b-1):
            b_n = int(self.n_v_batch/self.every_b)+1
            of_ = self.dataset_sizes['val']//self.every_b
            print(f'batch {b_n}/{of_} loss {self.val_lss/(b_n*self.every_b)}')
        self.n_v_batch += 1
        return {'val_loss' :loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def test_step(self, batch, batch_size):
        # print(f'val moce')
        x,y = batch
        logits = self.forward(x)
        #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits, y).cpu()
        return {'test_loss' : loss,'y_pred':preds.cpu(), 'y':y.cpu()}

    def validation_epoch_end(self, outputs):
        # print(f'end it')
        # OPTIONAL
        self.n_batch,self.n_v_batch= 0,0
        self.epoch += 1
        self.still_epoch = False
        self.trn_lss,self.val_lss = 0,0
        preds = torch.cat([x['y_pred'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        logs = {
                'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean(), 
                'val_acc':accuracy_score(preds,y),
                'val_f1_score':f1_score(preds, y, average='weighted'), 
                'val_pre_scr' : precision_score(preds, y, average='weighted'), 
                'val_recall_scr':recall_score(preds, y, average='weighted')
               }

        print(f"avg_val_loss:  {logs['val_loss']} avg_val_acc : {logs['val_acc']} val_f1_score : {logs['val_f1_score']} val_pre_scr : {logs['val_pre_scr']} val_recall_scr : {logs['val_recall_scr']}")
        if self.n_batch > 0 and self.still_epoch and logs['val_acc'] > self.best_acc and self.model_name:
          self.best_acc = logs['val_acc']
          model_info = {'state_dict_cnn' : self.state_dict(),'best_acc':self.best_acc,'epochs':self.total_epochs}
          torch.save(model_info, './models'+self.model_name+'_val.pt')
          torch.save(model_info, './models'+self.model_name+'_train.pt')
        return {'log': logs}

    def test_epoch_end(self, outputs):
        # print(f'maybe not')
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        preds = torch.cat([x['y_pred'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        
        logs = {'test_loss': avg_loss, 
                'test_acc':accuracy_score(preds,y),
                'test_f1_score':f1_score(preds, y, average='weighted'), 
                'test_pre_scr' : precision_score(preds, y, average='weighted'), 
                'test_recall_scr':recall_score(preds, y, average='weighted')
                }
                
        return {'log': logs}

    def prepare_data(self):
        # print(f'lets fix it')
        # download only    
        if self.train_flag:
            # print(f'in big')
            if  self.train_dataroot and self.train_datapath:
                # print(f'bit 1')
                train = torch_dset(csv_file=self.train_datapath, train=True, root=self.train_dataroot)

            if  (self.val_dataroot and self.val_datapath):
                # print(f'bit 2')
                val = torch_dset(csv_file=self.val_datapath, train=False, root=self.val_dataroot)

            elif (self.test_dataroot and self.test_datapath):
                # print(f'bit 3')
                val = torch_dset(csv_file=self.test_datapath, train=False, root=self.test_dataroot)

            self.dataloaders = {'train':DataLoader(train, batch_size=200, shuffle=True, num_workers=20, drop_last=True), 
                                'val':  DataLoader(val, batch_size=200, shuffle=False, num_workers=10, drop_last=True),
                                }
            self.dataset_sizes = {x: len(self.dataloaders[x]) for x in ['train', 'val']}
        else:
            # print(f'in smalll')
            #for test
            if (self.test_dataroot and self.test_datapath):
                # print(f'sim 1')
                test = torch_dset(csv_file=self.test_datapath, train=False, root=self.test_dataroot)
            elif (self.val_dataroot and self.val_datapath):
                # print(f'sim 2')
                test = torch_dset(csv_file=self.val_datapath, train=False, root=self.val_dataroot)
            self.dataloaders = {
                                'test':DataLoader(test, batch_size=200, shuffle=False, num_workers=10), 
                               }
            self.dataset_sizes = {x: len(self.dataloaders[x]) for x in ['test']}

    def train_dataloader(self):
        # print(f' tt1')
        #convert data to torch.FloatTensor
        #load the training dataset
        return self.dataloaders['train']

    def val_dataloader(self):
        # print(f'tt 2')
        return self.dataloaders['val']
    
    def test_dataloader(self):
        # print(f'tt 3')
        return self.dataloaders['test']