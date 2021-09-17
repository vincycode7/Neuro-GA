from torch.nn import functional as F
from GA_optim import *
from model import *

if __name__ == "__main__":

    #Initialize the GA Optimizer with the model class
    # fitness =     def my_loss(self, y_hat, y):
    #     return F.cross_entropy(y_hat, y)
    fitness_func = lambda y_hat_,y: F.cross_entropy(y_hat, y)
    model_instance = Custom_Model
    model_name = 'twoclasses_model1_ga_val.pt'
    test_dataroot='../'
    test_datapath='dataset/new_data4val4.csv'
    gen_algo = GA_optim(    
                            model_object=model_instance, 
                            Npop=1, 
                            num_generations=1,
                            epoch_per_model=1,
                            check_val_every_n_epoch = 1,
                            turn_off_backpropagation=False, 
                            num_parents_mating=1,
                            parent_selection = False,
                            crossover=False,
                            crossover_probability=0,
                            n_multi_crossover=2,
                            keep_parents=True, 
                            mutation=False,
                            mutation_probability=0,
                            mutation_num_chromes=1,
                            mutation_num_genes=1,
                            save_every_n_gen=1,
                            model_name=model_name,
                            to_gpu=1
                        )

    # gen_algo.from_state(path_to_state_dict=model_name)
    try:
        print(f'{model_name}')
        gen_algo.from_state(path_to_state_dict=model_name)
        print(f'loading Succesful')
    except Exception as e:
        print(f"Warning!!! Error Occured!!! while loading Model. Error is --> {e}")
        
    gen_algo.pop_tester(test_dataroot=test_dataroot,test_datapath=test_datapath)
    print(f"total fitness is : {gen_algo.fitness_function()}")

