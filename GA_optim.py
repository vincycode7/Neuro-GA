import torch
from model import Custom_Model
from torch import optim
from collections import OrderedDict
from pytorch_lightning import Trainer
from model import Custom_Model
import pytorch_lightning as pl
import numpy as np
import numpy,os,random,collections
from IPython.display import clear_output

#should Inherite from optim.Optimizer later
class GA_optim:
    def __init__(   self, 
                    model_object=None, 
                    num_generations=2,
                    num_parents_mating=5,
                    fitness_func=None,
                    Npop=10,  
                    epoch_per_model=1,
                    check_val_every_n_epoch = 3,
                    turn_off_backpropagation = False,
                    parent_selection = True, 
                    keep_parents=True, 
                    crossover = True,
                    crossover_probability=0.4,
                    n_multi_crossover=4,
                    mutation=True,
                    mutation_probability=0.05,
                    mutation_num_chromes=10,
                    mutation_num_genes=10,
                    save_every_n_gen=1,
                    model_name=None,
                    to_gpu=0
                    ):

        """
        Documentation
        =============

        This is a Genetic Algorithm class to optimize the performance of populations of models and selecting the best.
        
        Parameters
        ==========

        model_object(nn.Module object) : This parameter assumes training will be doing with pytorch and expects a pytorch model object.

        num_generations: Number of generations.

        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        Npop(int) : number of the model instance to be  created (Initial population of the parents).

        turn_of_backprobagation(bool) : This parameter determines if models or parents will be trained with back propagation before 
        Genetic algorithm is applied.

        True --> back propagation is not applied.

        False --> back propagation is applied.

        parent_selection(bool): if to select parents of not

        True --> To select
        Fasle --> To go with the current population

        keep_parents(bool) : to keep parent from current solution or not
        
            True --> to keep parents after cross over

            False --> To not keep parent after cross over

        crossover(bool): If to apply the cross over operator. 

            True --> yes apply cross over

            False --> no don't apply

        crossover_probability(float): The probability of selecting a solution for the crossover operation. If the solution probability 
        is <= crossover_probability, the solution is selected. The value must be between 0 and 1 inclusive.

        n_multi_crossover(int): number of n cross over

        mutation(bool): If to apply the mutation operator
            True --> yes apply the mutation operator

            False --> No, Don't apply the mutation operator

        mutation_probability(float): The probability of selecting a gene for the mutation operation. If the gene probability 
        is <= mutation_probability, the gene is selected. The value must be between 0 and 1 inclusive. If specified, then no 
        need for the parameters mutation_percent_genes, mutation_num_genes, random_mutation_min_val, and random_mutation_max_val.

        mutation_percent_genes: Percentage of genes to mutate which defaults to 10%. This parameter has no action if the parameter 
        mutation_num_genes exists.

        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then 
        no need for the parameter mutation_percent_genes.
        """
        assert epoch_per_model > 0 and type(epoch_per_model) == int, "epoch_per_model must be of type int and greater than 0"
        self.epoch_per_model = epoch_per_model
        assert check_val_every_n_epoch > 0 and type(check_val_every_n_epoch) == int, "check_val_every_n_epoch must be of type int and greater than 0"
        self.check_val_every_n_epoch = check_val_every_n_epoch
        assert type(to_gpu)==int
        self.to_gpu=to_gpu
        assert type(save_every_n_gen) == int, 'save every n gen should be int'
        self.save_every_n_gen = save_every_n_gen
        assert type(model_name) == str, 'model name should be a string'
        assert model_name.split('.')[-1] in ['pt','pth'], "model name format is invalid"
        self.model_name = model_name

        # assert type(model_object) == pl.LightningModule, 'model has to be of type pytorch-lighning'
        self.model_object = model_object
        # self.model = self.model_object()

        assert type(n_multi_crossover) and n_multi_crossover > 0, "The number of cross over point should be greater than 1"
        self.n_multi_crossover = n_multi_crossover
        assert type(Npop) == int, "population size has to be of type integer"
        self.Npop = Npop

        assert type(turn_off_backpropagation) == bool, "turn off backpropagation has to be of type bool."
        self.turn_off_backpropagation = turn_off_backpropagation

        #preserve model structure (Linear shape, None-Linear shape)
        self.None_linear_shapes = self.trace(state_dict_sample=self.model_object().state_dict())  #save linear and non-linear representation of weights

        # self.population = self.initial_population # A NumPy array holding the initial population.
        self.num_genes = sum([each[1][0] for each in self.None_linear_shapes])

        #create chromosomes
        # self.population = [self.model_object().state_dict() for each in range(self.Npop)] if from_state == False else self.from_state(path_to_state_dict=path_to_state_dict)
        self.population = [self.model_object().state_dict() for each in range(self.Npop)]
        self.pop_size = (len(self.population),self.num_genes) # The population size.
        self.tst_epoch_loss = []
        self.tst_epoch_acc = []
        self.fitness = [0 for each in range(self.Npop)]
        self.start_train_from = 0
        # self.trn_epoch_loss = []
        # self.trn_epoch_acc = []
        
        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type
        if crossover:
            self.crossover = self.multi_points_crossover
        else:
            self.crossover = None
        
        if crossover_probability == None:
            self.crossover_probability = None
        elif type(crossover_probability) in [int, float]:
            if crossover_probability >= 0 and crossover_probability <= 1:
                self.crossover_probability = crossover_probability
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'crossover_probability' parameter must be between 0 and 1 inclusive but {crossover_probability_value} found.".format(crossover_probability_value=crossover_probability))
        else:
            self.valid_parameters = False
            raise ValueError("Unexpected type for the 'crossover_probability' parameter. Float is expected by {crossover_probability_type} found.".format(crossover_probability_type=type(crossover_probability)))

        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        if (mutation):
            self.mutation = self.inversion_mutation
        else:
            self.mutation = None

        if mutation_probability == None and self.mutation:
            raise ValueError("Mutation is on, but no mutation probability provided.")
        elif type(mutation_probability) in [int, float]:
            if mutation_probability >= 0 and mutation_probability <= 1:
                self.mutation_probability = mutation_probability
            else:
                raise ValueError("The value assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but {mutation_probability_value} found.".format(mutation_probability_value=crossover_probability))
        else:
            raise ValueError("Unexpected type for the 'mutation_probability' parameter. Float is expected by {mutation_probability_type} found.".format(mutation_probability_type=type(mutation_probability)))

        if self.mutation:
            if (mutation_num_genes == None):
                raise ValueError("Genes to mutate is None when mutation is on")
            elif (mutation_num_genes <= 0):
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but {mutation_num_genes} found.\n".format(mutation_num_genes=mutation_num_genes))
            elif (mutation_num_genes > self.pop_size[1]):
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) ({mutation_num_genes}) cannot be greater than the number of genes ({num_genes}).\n".format(mutation_num_genes=mutation_num_genes, num_genes=self.num_genes))
            elif (type(mutation_num_genes) is not int):
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) must be a positive integer >= 1 but {mutation_num_genes} found.\n".format(mutation_num_genes=mutation_num_genes))
            elif type(mutation_num_genes) == float and mutation_num_genes < 0.0 and mutation_num_genes > 1.0:
                raise ValueError('float should be in between 0 and 1')
            elif type(mutation_num_genes) == float and mutation_num_genes > 0.0 and mutation_num_genes < 1.0:
                mutation_num_genes = numpy.ceil(self.pop_size[1]* mutation_num_genes)
            elif (mutation_num_chromes == None):
                raise ValueError("chromosomes to mutate is None when mutation is on")
            elif (mutation_num_chromes <= 0):
                raise ValueError("The number of selected chromosomes for mutation (mutation_num_chromes) cannot be <= 0 but {mutation_num_chromes} found.\n".format(mutation_num_chromes=mutation_num_chromes))
            elif (mutation_num_chromes > self.pop_size[0]):
                raise ValueError(f"The number of selected chromosomes for mutation (mutation_num_chromes) ({mutation_num_chromes}) cannot be greater than the number of genes ({self.pop_size[0]}).\n")
            elif (type(mutation_num_chromes) is not int):
                raise ValueError("The number of selected genes for mutation (mutation_num_chromes) must be a positive integer >= 1 but {mutation_num_chromes} found.\n".format(mutation_num_chromes=mutation_num_chromes))
            elif type(mutation_num_chromes) == float and mutation_num_chromes < 0.0 and mutation_num_chromes > 1.0:
                raise ValueError('float should be in between 0 and 1')
            elif type(mutation_num_chromes) == float and mutation_num_chromes > 0.0 and mutation_num_chromes < 1.0:
                mutation_num_chromes = numpy.ceil(self.pop_size[0]* mutation_num_chromes)
                
        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
        if parent_selection:
            self.select_parents = self.roulette_wheel_selection
            # Validating the number of parents to be selected for mating (num_parents_mating)
            if num_parents_mating <= 0:
                raise ValueError("The number of parents mating (num_parents_mating) parameter must be > 0 but {num_parents_mating} found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(num_parents_mating=num_parents_mating))

            # Validating the number of parents to be selected for mating: num_parents_mating
            if (num_parents_mating > self.pop_size[0]):
                raise ValueError(f"The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({self.pop_size[0]}) (i.e., num_parents_mating must always be <= sol_per_pop).\n")

            self.num_parents_mating = num_parents_mating
        else:
            self.select_parents = None
  
        # The number of completed generations.
        self.generations_completed = 0

        # Parameters of the genetic algorithm.
        self.num_generations = abs(num_generations)

        # Parameters of the mutation operation.
        self.mutation_num_chromes = mutation_num_chromes
        self.mutation_num_genes = mutation_num_genes
        self.trained_with_ga = False

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        # self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.

        # self.best_solution_generation = -1 # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.
        self.is_init = True
        self.best_solution_fitness = 0

    def trace(self, state_dict_sample=None):
        """
            Documentation
            =============

            trace model to get the weights names
            and save it for later decoding.

            Parameters
            ==========

            state_dict_sample :- a sample of the model state dict
        
        """
        return [(each, state_dict_sample[each].flatten().shape, state_dict_sample[each].shape) for each in state_dict_sample]
    
    def encode_param(self):
        """
            Docstring
            =========
            Convert all weights into a 1d array. 
            This helps in training both in GA
            optimization and in back-propagation optimization.

            Parameters
            ==========

            structure :-  should be a (tuple or list) of (tuple or list) 
            where each (tuple or list) in (list or tuple) has the
            structure 
            ('name of weight', size_of_weight_when_flattened, size_of_weight_when_not_flattened)

            params :- the pytorch model state dict
        """
        for idx in range(len(self.population)):
            self.population[idx] = torch.cat([self.population[idx][each].flatten().cpu() for each in self.population[idx]]).reshape(1,-1)
        self.population = torch.cat(self.population, axis=0).numpy()

    def decode_param(self):
        """
        

            Docstring
            =========
            Decode the 1d array into model weights. 
            This helps in training both in GA
            optimization and in back-propagation optimization.

            Parameters
            ==========

            structure :-  should be a (tuple or list) of (tuple or list) 
            where each (tuple or list) in (list or tuple) has the
            structure 
            ('name of weight', size_of_weight_when_flattened, size_of_weight_when_not_flattened)

            params :- the 1d pytorch array
        
        """
        all_states = []
        for idx in range(self.population.shape[0]):
            state_dict, keep_tab = OrderedDict(), 0
            for each in self.None_linear_shapes:
                state_dict[each[0]] = torch.Tensor(self.population[idx][keep_tab:keep_tab+each[1][0]].reshape(each[2])).cpu()
                keep_tab += each[1][0]
            all_states.append(state_dict)
        self.population = all_states

    def train(self,generation=5,max_gen_to_keep_parent=5):
        """
            where all training will take place

        """
        for each_gen in range(generation): #Loop for n generations
            print(f'Current Completed Generation -----> {self.generations_completed}')
            
            # train with back-prop if parameter is true
            if not self.turn_off_backpropagation:
                if not self.trained_with_ga:
                    self.backprop_train()
                self.trained_with_ga = True

            #Genetic algorithm starts(test Population Performance)
            self.ga_train(each_gen=each_gen, max_gen_to_keep_parent=max_gen_to_keep_parent)

            if self.select_parents: self.decode_param() # Decode population into state dict

            self.generations_completed += 1 # The generations_completed attribute holds the number of the last completed generation.
            self.save_state(name_of_state=self.model_name) #save normal train state after all gen
            
            _,curr_best_fit = self.best_solution() #picks the current best fitness after training
            if curr_best_fit >= self.best_solution_fitness: #save best states
                self.best_solution_fitness = curr_best_fit
                name,ext = self.model_name.split('.')
                model_name = name+'_val.'+ext
                self.save_state(name_of_state=model_name)
            self.reset_trainer_data()
        
    def reset_trainer_data(self):
        self.trained_with_ga = False
        self.start_train_from = 0
        
    def pop_tester(self):
        """
           Helps run test on population, to get their fitness
           
        """
        self.model = self.model_object(model_name=None,every_b=5000,best_acc=0,curr_ephochs=0,train_flag=False, test_dataroot='../', test_datapath='dataset/new_data4val4.csv')
        trainer = Trainer(  
                            max_epochs=self.epoch_per_model, 
                            check_val_every_n_epoch=self.check_val_every_n_epoch,
                            gpus=self.to_gpu,
                            reload_dataloaders_every_epoch=True,
                        )
        for idx in range(len(self.population)):
            
            #first load in state_dict
            self.model.load_state_dict(self.population[idx])
            result = trainer.test(self.model) #check for performance
            self.fitness_function(result[0], fit_idx=idx)  #saved the result of the population test into a variable
                
    def ga_train(self,each_gen, max_gen_to_keep_parent):
            
            #self.tester must be called to compute current test result for each epoch
            self.pop_tester()
            
            #use performance to select parents
            # Selecting the best parents in the population for mating.
            # self.best_solution()
            
            if self.select_parents != None:
                self.encode_param()
                self.select_parents(fitness=self.fitness, num_off=self.num_parents_mating)

            if self.crossover != None:
                if self.select_parents == None:
                    raise ValueError(" Select is None and crossover is True")
                if each_gen % max_gen_to_keep_parent != 0: #timer for how long parents can move to next gen
                    self.crossover(offspring_size=self.n_multi_crossover, keep_parents=True) 
                else:
                    self.crossover(offspring_size=self.n_multi_crossover, keep_parents=False)

            if self.mutation != None:
                if self.select_parents == None:
                    raise ValueError(" Select is not and crossover is True")
                self.mutation(n_gene_mutate=self.mutation_num_genes, num_chrms_mut=self.mutation_num_chromes)
                
    def backprop_train(self):
        #This is used to changed a part of the behavior of pytorch
        def set_random_port(self, force=False):
            """
            When running DDP NOT managed by SLURM, the ports might collide
            """
            # pick a random port first
            assert self.num_nodes == 1, 'random port can only be called from single node training'
            pid = os.getpid()
            rng1 = np.random.RandomState(pid)
            RANDOM_PORTS = rng1.randint(10000, 19999, 1)
            default_port = RANDOM_PORTS[-1]

            # when not forced, use the user port
            if not force:
                default_port = os.environ.get('MASTER_PORT', default_port)

            os.environ['MASTER_PORT'] = str(default_port)

        Trainer.set_random_port = set_random_port
        trainer = Trainer(  
                            max_epochs=self.epoch_per_model, 
                            check_val_every_n_epoch=self.check_val_every_n_epoch,
                            gpus=self.to_gpu,
                            reload_dataloaders_every_epoch=True,
                        )
                        
        train_start = self.start_train_from
        for idx in range(train_start, len(self.population)): #loop through n chromosomes
            print(f"On Population {idx} of {len(self.population)}")
            #create new instance of the model and load in the chromosomes state_dict
            self.model = self.model_object(model_name=None,every_b=10000,best_acc=0,curr_ephochs=0,train_flag=True,train_dataroot='../', train_datapath='dataset/new_data4train4.csv', val_dataroot='../', val_datapath='dataset/new_data4val4.csv')
            self.model.load_state_dict(self.population[idx].copy())

            #train with backpropagation
            trainer.fit(model=self.model)
            self.population[idx] = self.model.cpu().state_dict()
            self.start_train_from += 1
            self.save_state(name_of_state=self.model_name) #save train state after each pop train
            
    def save_state(self,name_of_state='states.pt'):
        torch.save(OrderedDict([
                                    ('model_object', self.model_object),
                                    ('population', self.population),
                                    ('pop_size', self.pop_size),
                                    ('prev_fitness', self.fitness),
                                    ('start_train_from', self.start_train_from),
                                    ('best_fitness', self.best_solution_fitness),
                                    ('gen_complete', self.generations_completed),
                                    ('test_loss', self.tst_epoch_loss),
                                    ('test_acc', self.tst_epoch_acc),
                                    ("num_genes",self.num_genes),
                                    ('None_linear_shapes', self.None_linear_shapes)
                                ]),
                                name_of_state)

    def from_state(self,path_to_state_dict=None):
        assert self.is_init,'There is no init yet'
        
        #import state
        try:
            dict_of_states = torch.load(path_to_state_dict)
        except Exception as e:
            raise ValueError(f"error {e} occured")
        
        # assert self.num_genes == dict_of_states["num_genes"], "None Linear Relationship not preserved"
        
        # self.num_genes = dict_of_states["self.num_genes"]
        self.None_linear_shapes = dict_of_states['None_linear_shapes']
        self.model = dict_of_states["model_object"]
        
        #load imported states
        try:
            assert len(dict_of_states['population']) == len(dict_of_states['prev_fitness']), "population and fitness list has to be the same length"
            if len(self.population) > len(dict_of_states['population']):
                self.population[:len(dict_of_states['population'])] = dict_of_states['population']
                self.fitness[:len(dict_of_states['population'])] = dict_of_states['prev_fitness']
                print("warning -- new population length is {len(self.population)} and previous was {len(dict_of_states['population'])}")
            elif len(self.population) < len(dict_of_states['population']):
                fit_np = np.array(dict_of_states['prev_fitness']).argsort()
                self.fitness = [dict_of_states['prev_fitness'][idx] for idx in fit_np][-len(self.population):]
                self.population = [dict_of_states['population'][idx] for idx in fit_np][-len(self.population):]
                print("warning -- new population length is {len(self.population)} and previous was {len(dict_of_states['population'])}")
            else:
                self.population = dict_of_states['population']
                self.fitness = dict_of_states['prev_fitness']
                
        except Exception as e:
            raise ValueError(f"Error -- {e} occured")
        
        # self.pop_size = dict_of_states['pop_size']
        self.model_object = dict_of_states['model_object']
        
        try:
            if dict_of_states['start_train_from'] < (self.pop_size[0]-1):
                self.start_train_from = dict_of_states['start_train_from']
        except:
            print(f"warning -- start_train_from is not found in state")
            
        try:
            self.best_solution_fitness = dict_of_states['best_fitness']
        except:
            print(f"warning -- best_solution_fitness is not found in state")
            
        try:
            self.generations_completed = dict_of_states['gen_complete']
        except:
            print(f"warning -- generation_completed is not found in state")
            
        # try:
        #     self.best_solution_chrm = dict_of_states['best_chromosome']
        # except:
        #     print(f"warning -- best_solution is not found in state")
            
        try:
            self.tst_epoch_loss = ['test_loss']
        except:
            print(f"warning -- tst_epoch_loss is not found in state")
            
        try:
            self.tst_epoch_acc  =  dict_of_states['test_acc']
        except:
            print(f"warning -- tst_epoch_acc is not found in state")
        
    def roulette_wheel_selection(self, fitness,num_off=3):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_off: The number of parents to be selected.
        It returns an array of the selected parents.
        """
        assert num_off <= len(fitness), 'numbers parents to select is greater than available population'
        max_ = numpy.array(sum(fitness))
        selection_probs = fitness/max_
        track_dominant = len(fitness) - list(selection_probs).count(0)

        new = []
        while len(new) != num_off:
            if track_dominant > 0:
                try:
                    hold = np.random.choice(len(fitness), p=selection_probs)
                except:
                    hold = np.random.choice(len(fitness))
                    
                if selection_probs[hold] != 0 and hold not in new:  
                    track_dominant -= 1
            else:
                hold = np.random.choice(len(fitness))
            if hold not in new:
                new.append(hold)
        self.population = self.population[new]

    def multi_points_crossover(self,offspring_size=4, keep_parents=False):

        """
        Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """
        if keep_parents:
            rem_off = self.pop_size[0] - len(self.population)
            num_of_iter = round(rem_off/2) if rem_off > 1 else 2
        else:
            rem_off = self.pop_size[0]
            num_of_iter = round(rem_off/2)

        offsprings = []
        for each in range(num_of_iter):
        
            #Select the parents to go for crossover
            while True:
                two_parents = list(numpy.random.randint(low=0, high=self.population.shape[0], size=2))
                if self.population.shape[0] == 1:
                    break

                if two_parents[0] != two_parents[1]:
                    break

            len_ = self.num_genes-2 #no of genes in a chrom - 2
            interval = int(len_/offspring_size)-1

            assert offspring_size < len_, "n points greater than length of chromosones"

            rang = numpy.round(numpy.linspace(1,len_,offspring_size+1))
            rang = [numpy.random.randint(low=rang[each], high=rang[each+1],size=1)[0] for each in range(len(rang)-1)]
            rang_ = []

            #Pick the ranges that splits the chrms in to n 
            for each in rang:
                rang_.append([0,each]) if len(rang_) == 0 else rang_.append([rang_[-1][1]+1,each])
            rang_.append([rang_[-1][1]+1, len_+2])
            del rang

            #pick the range to swap
            # num_of_swap = len(rang_)- offspring_size
            num_of_swap = offspring_size-1
            # swaps = list(numpy.random.randint(low=0, high=len(rang_), size=num_of_swap))
            swaps = []
            while True:
                pick = numpy.random.randint(low=0, high=len(rang_), size=1)[0]
                if pick not in swaps:
                    swaps.append(pick)
                if len(swaps) == num_of_swap:
                    break

            #Fill 1st offspring
            off0 = self.population[two_parents[0]].copy()

            #keep the swaps
            keep = [off0[rang_[each][0]:rang_[each][1]].copy() for each in swaps]

            for idx,each in enumerate(swaps):
                off0[rang_[each][0]:rang_[each][1]] = self.population[two_parents[1]][rang_[each][0]:rang_[each][1]].copy()

            #Fill 2nd Offstring
            off1 = self.population[two_parents[1]].copy()
            for idx, each in enumerate(swaps):
                off1[rang_[each][0]:rang_[each][1]] =  keep[idx]

            offsprings.append(off0.reshape((1,-1))) if len(offsprings) < rem_off else ''
            offsprings.append(off1.reshape((1,-1))) if len(offsprings) < rem_off else ''

        if keep_parents:
            offsprings = np.concatenate(offsprings)
            self.population = np.concatenate([self.population, offsprings])
        else:
            self.population = np.concatenate(offsprings)
        del offsprings

    def inversion_mutation(self,n_gene_mutate=10, num_chrms_mut=0.8):

        """
        Applies the inversion mutation which selects a subset of genes and inverts them.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        if type(n_gene_mutate) == int:
            pass
        elif type(n_gene_mutate) == float:
            n_gene_mutate = numpy.ceil(self.population.shape[1]*n_gene_mutate)

        if type(num_chrms_mut) == int:
            pass
        elif type(num_chrms_mut) == float:
            num_chrms_mu = numpy.ceil(self.population.shape[0]*num_chrms_mut)

        assert num_chrms_mut < self.population.shape[0], "number of chromosones to mutate is greater than number of chromosomes in the solution"
        assert n_gene_mutate < self.population.shape[1], "number of genes to mutate in a chromoosome is greater than number of genes in solution chromosomes"

        for _ in range(int(num_chrms_mut)):
            idx = numpy.random.randint(low=0, high=numpy.ceil(self.population.shape[0]), size=1)[0]
            mut_gen = numpy.random.sample()
            if mut_gen > (1-self.mutation_probability):
                for _ in range(int(n_gene_mutate)):
                    mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(self.population.shape[1]), size=1)[0]
                    coin = numpy.random.randint(0,2)
                    low = mutation_gene1 if coin else 0
                    mutation_gene2 = numpy.random.randint(low=low, high=numpy.ceil(self.population.shape[1]), size=1)[0]

                    if mutation_gene1 != mutation_gene2:
                        keep = self.population[idx,mutation_gene1]
                        self.population[idx, mutation_gene1] = self.population[idx, mutation_gene2]
                        self.population[idx, mutation_gene2] = keep

    def fitness_function(self, result = None, fit_idx=None):

        """  Calculate the fitness of each model after n-epoch or just return the already calculated fitness"""
        if result and fit_idx:
            self.tst_epoch_loss.append(result['test_loss'])
            result = (result['test_acc']+result['test_f1_score']+result['test_pre_scr']+result['test_recall_scr'])/4
            self.tst_epoch_acc.append(result)
            self.fitness[fit_idx] = result
        else:
            return self.fitness
            
    def best_solution(self):

        """
        Returns the index and fitness of the best chromosome in a population
        """
        
        #gets the list of all the fitness
        fitness = self.fitness_function()
        
        assert len(fitness) == self.Npop, "size of pop should be of the same length as fitness array"

        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
        try:
            print(f"good return --> {best_match_idx} -- {fitness[best_match_idx]}")
            return best_match_idx,fitness[best_match_idx]
        except:
            print('Error occured somewhere while picking best in best_solution function')
        print(f"bad return --> {best_match_idx} -- {fitness[best_match_idx]}")
        return 0