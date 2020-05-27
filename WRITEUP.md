Write-UP For ProJect

[Intro].
This Project Is one that aims to explore genetic algorithm and it's application in optimizing neural network for better performance. In this project the main field the two algorithms will be applied to is the *Custom field*, where goods are been checked for any fraudulent act. The project aims to achieve success in training a model that can successfully Identify fraudulent act in the process of importing goods thereby reducing time allocated to checking goods by narrowing search to goods that really call for concern. 

[The Dataset].
The dataset Used For this Project was gotten from custom and it is a collection of data of all the official daily records from 2018 to 2019. The Dataset has 21 features But out of this features only 13 will be relivant for the Project. The features are :-

                                    > 1. Importer | 'IMPORTER_NAME',
                                    > 2. Declarant | 'DECLARANT_NAME',
                                    > 3. Country of Origin | 'CTY_ORIGIN',
                                    > 4. Commodities | 'HS_DESC',
                                    > 5. Item price | 'ITM_PRICE',
                                    > 6. Mode of tramsport | 'MODE_OF_TRANSPORT',
                                    > 7. Number of items | 'ITM_NUM',
                                    > 8. Gross Mass | 'GROSS_MASS',
                                    > 9. Net Mass | 'NET_MASS',
                                    > 10. Tax amount | 'TOTAL_TAX',
                                    > 11. Invoice amount | 'INVOICE_AMT',
                                    > 12. Quantity | 'QTY',
                                    > 13. Statistical value | 'STATISTICAL_VALUE'.

5 Discrete Feature(s) --> (CTY_ORIGIN, MODE_OF_TRANSPORT, QTY, *ITM_NUM will be splitted into two features first  feature will be the current_quantity, second will be the total_quantity*). 6 continous Feature(s) --> (GROSS_MASS, NET_MASS, ITM_PRICE, STATISTICAL_VALUE	, TOTAL_TAX	, INVOICE_AMT). 4 string Feature(s) -->(IMPORTER_NAME, DECLARANT_NAME, HS_DESC, INSPECTION_ACT).




[The Algorithm].
So, The way the algorithm is going to work is, the user will have the option of either training the algorithm with only Gradient Descent as the optimizer or Only Genetic Algorithm as the Optimizer or Training with both Gradient Descent and Genetic Algorithm. Since GD is not an evolution algorithm, if training with only GD is enabled, only GD will be used to optimize the individual instances of the model seperately, On the other hand since GA is an evolution algorithm it will be used to collectively optimize a fixed number of populations together, this population otherwise known as chromosomes in GA or weight of an instance In Neural Networks after some genration are expected to help each other get the work done using some special operators.


            << About the Artificial neural network (https://en.wikipedia.org/wiki/Artificial_neural_network)>>
Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge of cats, for example, that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the examples that they process.

An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron that receives a signal then processes it and can signal neurons connected to it.

In ANN implementations, the "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.

The original goal of the ANN approach was to solve problems in the same way that a human brain would. But over time, attention moved to performing specific tasks, leading to deviations from biology. ANNs have been used on a variety of tasks, including computer vision, speech recognition, machine translation, social network filtering, playing board and video games, medical diagnosis, and even in activities that have traditionally been considered as reserved to humans, like painting.


            <<  About Genetic Algorithm (https://www.sciencedirect.com/topics/engineering/genetic-algorithm)>>
Genetic algorithms (GA) like neural networks are biologically inspired and represent a new computational model having its roots in evolutionary sciences. Usually GAs represent an optimization procedure in a binary search space, and unlike traditional hill climbers they do not evaluate and improve a single solution but a set of solutions or hypotheses, a so-called population. The GAs produce successor hypotheses by mutation and recombination of the best currently known hypotheses. Thus, at each iteration a part of the current population is replaced by offspring of the most fit hypotheses. In other words, a space of candidate hypotheses is searched in order to identify the best hypothesis, which is defined as the optimization of a given numerical measure, the so-called hypothesis fitness. Consider the case of function approximation based on given input-output samples: The fitness is the accuracy of the hypothesis (solution) over the training set.

The strength of this parallel process is enhanced by the mechanics of population modification, making GAs adequate candidates even for NP-hard problems. Mathematically, they are function optimizers and they encode a potential solution based on chromosome-like data structures. The critical information is preserved by applying recombination operators to these structures .


[OUR Method].

                                                    << Initialize Population >>

So basically, Just like every other genetic algorithm applications, n populations chromosomes will be initialized,where n is the number of individual or chromosomes to be randomly created. However In This case the chromosomes are not going to be 0s and 1s but the weights of the n instance of the ANN archtecture i.e, we are creating n number of models with the same architecture for the purpose of finding the optimum solution. To archieve a Linear weight, the weight of the various models will be flattened into a 1d array, where weight order is retained from the input layer to the output layer.

                                                        << The loop >>
                                *The loop* are it implices is the repetition of the following steps

                                                **forward and backward propagation**
In this step, we will pass batch of data into each of the network in a forward approach, after which fitness function is applied to calcution *loss*, *accuracy*, *f1score*. After the forward pass the first stage of optimization is done with Gradient Descent. 


                                                        ** Fitness Funtion **
The first optimization using GD is immedietly followed by a GA optimization using the *loss, accuracy and f1score* as it's fitness funtion. The fitness value is calculated by normalizing each of the values and find their mean.

== implementation example. ==

--fitness for loss--
given loss of [0.1,0.35,0.7]

(1) we normalize by first summing up the values
0.1+0.35+0.7 = 1.15

(2) then we get the normalized value for each
 -- 1-(0.1÷1.15) = 0.913
 -- 1-(0.35÷1.15) = 0.6957
 -- 1-(0.7÷1.15) = 0.3913

0.913+0.6957+0.3913 = 2





                                                        ** Selection Operator (roulette-wheel selection)**

                                                        ** Cross-Over Operator **

                                                        ** Mutation Operator **

                                                             ** Elitism **

[What is been optimuzed]

