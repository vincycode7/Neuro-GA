import torch
from model import custom_model
from torch import optim


#should Inherite from optim.Optimizer later
class GA_optim:
    def __init__(self, model_object, Npop=10,  Pc=0.5, Pm=0.05):
        self.model_object = model_object
        self.Npop = 
        self.chrms = [self.model_object() for each in range(self.Npop)]

    def trace(self, model=None):
        return [[{}]]
    
    def encode_param(self, params=None):
        return torch.Tensor([])

    def decode_param(self, params=[None], trace=None):
        return state_dict

    def train(self,model):
        model.train()
        model.forward()
        return model

    def back_prop(self, loss):
        loss.backward()

    def load_weight(self, model, state_dict):
        return model

    def def fitness(self, chroms = None):
        pass

    def selection(self, fit_fun=None,size=None):
        pass

    def crossover(self, prt_chroms = None, rlow=(0,6), rhigh=(8,13), mutate=True):
        pass

    def mutation(self, chrom=None, rlow=(0.09,0.6), rhigh=(0.5,1.0)):
        pass


    def elitism(self, prnt, chld):
        pass
