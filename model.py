
#model creation
# import libraries
# !pip install pytorch-lightning
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
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
from torch.nn import Sequential,Dropout, Linear, Identity, Conv2d
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import SGD
import os

#building the pytorch lighting class Model
class Custom_Model(pl.LightningModule):
    def __init__(self,both=None,pos=None,neg=None,model_name=None,every_b=3,best_acc=0):
        """
          self.model_name --> This the name You would Wish to save your model as
          self.n_batch --> This is the amount of batch that has been passed through the current Epoch in the train step
          self.n_v_batch --> This is the amount of batch that has been passed through the current Epoch in the val step
          self.every_b --> The Amount Of batch That should be passed Before running computation is displayed
          self.trn_acc --> This running accuracy for the train step
          self.trn_lss --> This is the running loss for the train step
          self.val_acc --> This is the running accuracy for the val step
          self.val_lss --> This is the running loss for the val step
          self.best_acc --> This best computed accuracy while batches run through a certain epoch
          self.epoch --> How many epoches there is to run through
          self.still_epoch --> This is a boolean value to indicate if we are still in the current epoch
        """

        #Define Hyperparameters
        super(Custom_Model, self).__init__()
        self.prepare_data()
        self.n_batch,self.n_v_batch,self.trn_acc,self.trn_lss, self.val_acc, self.val_lss= 0,0,0,0,0,0
        self.model_name,self.every_b = model_name, every_b
        self.best_acc,self.epoch,self.still_epoch = best_acc,0,False

        #Define Model architecture
        self.fc1 = nn.Linear(10,8)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8,5)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(5,2)
        self.sig = nn.Sigmoid()
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.sig(self.fc3(x))
        return x

    def create_network(json_config=None):
        return 
        

    def configure_optimizers(self):
        return SGD([{'params' : self.parameters(), 'lr':0.0001}],
                            lr=0.0001, momentum=0.9)

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y[:,0])

    def training_step(self, batch, batch_size):
        if self.n_batch == 0 and not self.still_epoch:
          with open('model_out.txt',mode='w') as f:
            clear_output()
            f.write(f'<============= Epoch {self.epoch} ==============>\n')
          self.still_epoch = True
        x,y = batch
        
        #forward pass
        logits = self.forward(x)
        
        #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits.cpu(), y.cpu())
        acc = torch.tensor(accuracy_score(preds.cpu(), y.cpu()))

        #add up train loss and accuracy
        self.trn_lss += loss
        self.trn_acc += acc
        
        #save model state
        model_info = {'state_dict_cnn' : self.state_dict(),'best_acc':self.best_acc}
        torch.save(model_info, './models'+self.model_name+'_train.pt')
        
        #print out something if it is on every third batch
        if self.n_batch%self.every_b==(self.every_b-1):
            b_n = int(self.n_batch/self.every_b)+1
            of_ = self.dataset_sizes['train']//self.every_b
            with open('model_out.txt',mode='a') as f:
              f.write(f'batch {b_n}/{of_} loss {self.trn_lss/(b_n*self.every_b)} acc {self.trn_acc/(b_n*self.every_b)}\n')
        self.n_batch+=1
        return {'loss' : loss}
        # return loss (also works)

    def validation_step(self, batch, batch_size):
        x,y = batch
        # x.to(self.device)
        # self.to(self.device)
        logits = self.forward(x)
        
        #get predictions and loss
        _, preds = torch.max(logits, 1)
        loss = self.my_loss(logits.cpu(), y.cpu())
        acc = torch.tensor(accuracy_score(preds.cpu(), y.cpu()))
        #add up train loss and accuracy
        self.val_lss += loss
        self.val_acc += acc
        
        #print out something if it is on every third batch
        if self.n_v_batch%self.every_b==(self.every_b-1):
            b_n = int(self.n_v_batch/self.every_b)+1
            of_ = self.dataset_sizes['val']//self.every_b
            with open('model_out.txt',mode='a') as f:
              f.write(f'batch {b_n}/{of_} loss {self.val_lss/(b_n*self.every_b)} acc {self.val_acc/(b_n*self.every_b)}\n')
        self.n_v_batch += 1
        return {'val_loss' :loss, 'val_acc':acc}

    def test_step(self, batch, batch_size):
        x,y = batch
        # x.to(self.device)
        # self.to(self.device)
        logits = self.forward(x)
        _, preds = torch.max(logits, 1)
        acc = torch.tensor(accuracy_score(preds.cpu(), y.cpu()))
        return {'test_loss' : self.my_loss(logits, y), 'test_acc':acc}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        self.n_batch,self.n_v_batch= 0,0
        self.epoch += 1
        self.still_epoch = False
        self.trn_lss,self.trn_acc,self.val_lss,self.val_acc = 0,0,0,0
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print(f'avg_val_loss:  {avg_loss} avg_val_acc : {avg_acc}')
        if avg_acc > self.best_acc:
          self.best_acc = avg_acc
          model_info = {'state_dict_cnn' : self.state_dict(),'best_acc':self.best_acc}
          torch.save(model_info, './models'+self.model_name+'_val.pt')
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs,'avg_val_acc':avg_acc}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs,'avg_test_acc':avg_acc}

    

    def prepare_data(self):
      #initialize the train and test data instance
    train, val, test = Dataloader.load_csv_rel(both=self.both)

    self.dataloaders = {'train':DataLoader(train, batch_size=100, shuffle=True, num_workers=5), 
                    'val':DataLoader(val, batch_size=70, shuffle=False, num_workers=1),
                    'test':DataLoader(test, batch_size=20, shuffle=False, num_workers=1)
               }
    self.dataset_sizes = {x: len(self.dataloaders[x]) for x in ['train', 'val','test']}

    def train_dataloader(self):
        #convert data to torch.FloatTensor
        #load the training dataset
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']