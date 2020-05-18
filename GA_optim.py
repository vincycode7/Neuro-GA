import torch
from model import custom_model
from torch import optim


#should Inherite from optim.Optimizer later
class GA_optim:
    def __init__(self, model_object, Npop=10,  Pc=0.5, Pm=0.05):
        self.model_object = model_object
        self.Npop = Npop
        self.chrms = [self.model_object() for each in range(self.Npop)]
        self.Pc = Pc
        self.Pm = Pm

    def trace(self, model=None):
        return [[{}]]
    
    def encode_param(self, params=None):
        return torch.Tensor([])

    def decode_param(self, params=[None], trace=None):
        pass

    def train(self,epoch=5):
        new_tens = torch.Tensor([])
        # print(torch.cat([torch.cat([new_tens, torch.Tensor([1,2,4])]), torch.Tensor([1,2,4])]))
        for name, param in self.model_object().named_parameters():
        #     # print(name,param)
            # print(f'before {param.data.shape} -- after {param.data.flatten().detach().shape}')
            new_tens = torch.cat([new_tens, param.data.flatten().detach()])
            # print(f' before {param.data}')
            # j = param.data
            # # print(param.data)
            # print(f'{j.requires_grad}')
            # k = torch.FloatTensor(param.data.shape)
            # param.data = k
            # print(param.requires_grad)
            # print(f' after {param.data}')
        # print(f'checking model out {gen_algo.model_object().state_dict().keys()}')
        print(new_tens.shape)
        pass

    def back_prop(self, loss):
        pass

    def load_weight(self, model, state_dict):
        return model

    def fitness(self, chroms = None):
        pass

    def selection(self, fit_fun=None,size=None):
        pass

    def crossover(self, prt_chroms = None, rlow=(0,6), rhigh=(8,13), mutate=True):
        pass

    def mutation(self, chrom=None, rlow=(0.09,0.6), rhigh=(0.5,1.0)):
        pass


    def elitism(self, prnt, chld):
        pass

    