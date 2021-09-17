| Details                   |                |
| ------------------------- | -------------- |
| Programming Language:     | Python 3.**6** |
| pytorch version:          | >=1.5.1        |
| pytorch lightning version | >=0.8.5        |
| Models Required:          | custom model   |
| OS                        | Ubuntu 18      |

#Short introduction to the project
This is a project is based on been able to detect complaint goods and non compliant goods during importation. This project was also incorporated genetic algorithm during training.

## Project Set Up and Installation on Ubuntu

_TODO:_ Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

[step 1]
The project heavily depends on python as it's the language used to run the whole program so, the first thing to do is to install python before proceeding.

[step 2]
install pip, pip is a python package that helps install other python packages.

[step 3]
use pip to install the rest of the required package. To do this, make sure your current directory structure is this from the terminal or command line.
.
├── dataset.csv
├── label_fea_onehotencoder4.joblib
├──normalizer4.joblib
├──standardizer4.joblib
├──string_encoder4.joblib
├── models-without-ga.pt
├── models-with-ga.pt
|── **init**.py
|── .gitignore
|── DR_note.ipynb
|── process_data.py
├── README.md
├── requirements.txt
|── test.py
|── train.py
|── WRITEUP.md

type,
**pip install -r requirements.txt** in the terminal

this command will install other required package for the project.

#model
The model architecture for this model is a custom model and a simple ANN architecture. To run the training from scratch the model.py file contains the architecture, but to train using the pre-trained version, a link to download the pretrained version will be attached below.

#how to use the process_data.py script
the feature in which the process_data.py file works with are in this format
#all features
all_features= [ 'IMPORTER_NAME', 'DECLARANT_NAME', 'CTY_ORIGIN', 'MODE_OF_TRANSPORT',
'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS','NET_MASS', 'ITM_PRICE',
'STATISTICAL_VALUE', 'TOTAL_TAX', 'INVOICE_AMT',
]
#target
target = ['target']

all the values in both `all_features` and `target` must be present, the `target` column can contain all null because after running the script the values will be filled again.

#demo run process_data.py
use the python3 process_data.py

#Write on how to use main.py script to train a model or continue from previous training

[train from scratch]
to train a model you simply run `python main.py` but before you do make sure to edit the `model_name` variable in the main.py file to specify the name to save the checkpoint as. by default name is 'twoclasses_model1_ga.pt'

[continue training from a checkpoint]
to continue training from a check point simply run `python main.py` but before you do make sure to edit the `model_name` variable in the main.py file to specify the path of the checkpoint. by default name is 'twoclasses_model1_ga.pt'

**note**:- to understand how to to finetune parameters going into the GA_optim class read the doc in the GA_optim class.

_TODO_ Write on how to use the test.py script
to test the a model's performance simply run `python test.py` but before you do make sure to set model_name(name/path to model to test default is 'twoclasses_model1_ga.pt'), test_dataroot(the root directory to the test file from you current working directory default is '../' which means moving a step outside the current working directory), test_datapath(the name/path of the csv test file from the root directory defualt is 'dataset/new_data4val4.csv')
