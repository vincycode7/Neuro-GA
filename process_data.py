from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from joblib import dump, load
import pandas as pd
import numpy as np
import time,re

# Drop duplicates
class Drop_Similars(BaseEstimator, TransformerMixin):
    def __init__(self,to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        """ This is a class to concat multiple similar pandas dataframe together
            by column.
        
        """
        
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X.drop_duplicates()

# Class To Replace Some Characters
class Replace_Values(BaseEstimator, TransformerMixin):
    def __init__(self,to_replace=['\n', ':-', ':', '-', ',', '.', ' \n', ' \n '], value=[''], inplace=False, limit=None, regex=False, method='pad'):
        """ This is a class to concat multiple similar pandas dataframe together
            by column.
        
        """
        self.to_replace=to_replace, 
        self.value=value,
        self.inplace=inplace, 
        self.limit=limit
        self.regex=regex,
        
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X.replace(self.to_replace,self.value,self.inplace,self.limit,self.regex)

#Class to concat Multiple dataset if available
class Concat_Dsets(BaseEstimator, TransformerMixin):
    def __init__(self,features=None):
        """ This is a class to concat multiple similar pandas dataframe together
            by column.
        
        """

    def fit(self,data_frames,y=None):
        return self

    def transform(self,data_frames,y=None):
        """
            data_frames : a list of dataframe(s)
        """
        return pd.concat(data_frames, ignore_index=True)

#Class to Pick interested Feature and Throw an Error if any Not Available
class All_Features(BaseEstimator, TransformerMixin):
    def __init__(self,features=None):
        """ This is a class to pick interested features from a pandas dataframe
            
            feature(type list or tuple):- this parameter is used to specify the list
            that should be selected, default ['IMPORTER_NAME', 'DECLARANT_NAME', 
            'OFFICE_COD', 'OFFICE_NAME', 'CTY_ORIGIN','MODE_OF_TRANSPORT', 'HSCODE',
            'HS_DESC', 'ITM_NUM', 'QTY', 'GROSS_MASS', 'NET_MASS', 'ITM_PRICE',
            'STATISTICAL_VALUE', 'TOTAL_TAX', 'INVOICE_AMT', 'INSPECTION_ACT']
            is used if feature is None.
        
        """

        default = ['IMPORTER_NAME', 'DECLARANT_NAME', 'OFFICE_COD', 'OFFICE_NAME', 'CTY_ORIGIN',
                   'MODE_OF_TRANSPORT', 'HSCODE', 'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS',
                   'NET_MASS', 'ITM_PRICE', 'STATISTICAL_VALUE', 'TOTAL_TAX', 'INVOICE_AMT',
                   'INSPECTION_ACT']

        self.features = default if not features else features
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X[self.features].copy()


#Encode categorical features
class Encode_Feature_Label(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        encoding N features and filling missing values
        too
    """
    
    def __init__(self, all_features=['size_D', 'duration_D', 'app', 'har', 
                                    'dba', 'ifc', 'source', 'nlan',
                                    'telonuse', 't01', 't02', 't03', 
                                    't04', 't05', 't06', 't07', 't08',
                                    't09', 't10', 't11', 't12', 't13', 
                                    't14', 't15'],  
                        
                        to_encode=[ 'OFFICE_COD', 'CTY_ORIGIN', 
                                    'MODE_OF_TRANSPORT', 'HSCODE']):
    
        #Read in data
        self.features = all_features
        self.to_encode = to_encode

    def fit(self,X):
        #check if features are present
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp

        self.all_encode = {each_feature : LabelEncoder().fit(X[each_feature]) for each_feature in self.to_encode}
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset
        """
        #check if features are present
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
            
        #Replace Labels with numerical values
        for each_feature in self.to_encode:
            X[each_feature] = self.all_encode[each_feature].transform(X[each_feature])
            classes_ = self.all_encode[each_feature].classes_
            none_index = np.where(classes_ == 'NaN')[0]
            if none_index.shape[0] >= 1:
                none_index = int(none_index)
                X[each_feature].replace(none_index,np.nan,inplace=True)
        return X

#Encode String-like features
class String_Encoder(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        encoding N string like features.
    """
    
    def __init__(self, all_features=['IMPORTER_NAME', 'DECLARANT_NAME', 
                                    'OFFICE_NAME', 'CTY_ORIGIN', 'MODE_OF_TRANSPORT', 
                                    'HSCODE', 'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS',
                                    'NET_MASS', 'ITM_PRICE', 'STATISTICAL_VALUE', 
                                    'TOTAL_TAX', 'INVOICE_AMT', 'INSPECTION_ACT'],  
                        
                        to_encode=['INSPECTION_ACT']):
    
        #Read in data
        self.features = all_features
        self.to_encode = to_encode

    def fit(self,X):
        #check if features are present
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp

        self.all_encode = {each_feature : CountVectorizer().fit(X[each_feature]) for each_feature in self.to_encode}
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset
        """
        #check if features are present
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
            
        #Replace Labels with numerical values
        for each_feature in self.to_encode:
            X[each_feature] = self.all_encode[each_feature].transform(X[each_feature])
        return X

#Fill missing values
class Fill_Empty_Spaces(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        Filling Missing Values with Standard Values That
        Represents Missing Values, e.g numpy.nan.
    """
    
    def __init__(self, all_features=[   'TeamExp', 'ManagerExp', 'YearEnd','Length', 
                                        'Transactions','Entities','PointsAdjust', 
                                        'Envergure','PointsNonAjust', 'Langage','Effort'],

                        find_in=[   'TeamExp', 'ManagerExp', 'YearEnd','Length', 
                                    'Transactions','Entities','PointsAdjust', 
                                    'Envergure','PointsNonAjust', 'Langage','Effort'],
                                        
                        find=None,
                        with_=None
                        ):
    
        #Read in data
        self.features = all_features
        self.find_in = find_in
        self.find = ['?','? ',' ?',' ? ','',' ','-',None,'None','none','Null','null',np.nan] if not find else find
        self.with_ = np.nan if not with_ else with_

    def fit(self,X):
        return self #do nothing
    def transform(self,X):
        """
            Work on the dataset
        """
        
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
            
        #Replace Missing Value With Recognized Missing Value
        X[self.find_in] = X[self.find_in].replace(self.find,self.with_)
        return X

#Label Instances
class Labeller(BaseEstimator, TransformerMixin):
    def __init__(self,features=None):
        """ This is a class to pick interested features from a pandas dataframe
            
            feature(type list or tuple):- this parameter is used to specify the list
            that should be selected, default ['IMPORTER_NAME', 'DECLARANT_NAME', 
            'OFFICE_COD', 'OFFICE_NAME', 'CTY_ORIGIN','MODE_OF_TRANSPORT', 'HSCODE',
            'HS_DESC', 'ITM_NUM', 'QTY', 'GROSS_MASS', 'NET_MASS', 'ITM_PRICE',
            'STATISTICAL_VALUE', 'TOTAL_TAX', 'INVOICE_AMT', 'INSPECTION_ACT']
            is used if feature is None.
        
        """

        default = ['IMPORTER_NAME', 'DECLARANT_NAME', 'CTY_ORIGIN',
                   'MODE_OF_TRANSPORT', 'HSCODE', 'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS',
                   'NET_MASS', 'ITM_PRICE', 'STATISTICAL_VALUE', 'TOTAL_TAX', 'INVOICE_AMT',
                   'INSPECTION_ACT']

        self.features = default if not features else features
    def fit(self,X,y=None):
        assert list(X.columns) == self.features
        return self

    def transform(self,X,y=None):
        X['target'] = 1
        return self.label_datatarget(X)

    @staticmethod
    def label_datatarget(X=None):
        """ function used to assign label value to data, 1 for compliant 
            and 0 for non compliant 
        """

        ones= [ 'RELEASE AUTHORISED',' RELEASE AUTHORISED.',',RELEASE AUTHORISED', 'RELEASE AUTHORISED.','RELSD BY',
                'RLEASAE',' LEASED ','RELEASDE BASED ON','CORRECTLY CLASSIFIED',
                'VERIFIED','RE- LEASED','AUTH',
                'RELSD  BASED',
                ' IN COMPLIANCE','RLS AUTHORISED',
                'RLDS BY',
                'RELASED','RELAESED','RELEASAED',
                'RELAESED',
                ' RELD ','AUTHORIZED','AUTHORISE.',
                'RLEASED','RELS BSD ON',
                'RLEASED.','.RLS AUTHORISED','RLEASE RECOMMENDED','REL AUTHO','RELSD BASED ON EXM.',
                'RLS AUTHORISED','ELEASE AUTHORISED','RLS AUTHORISED.','RELEASA AUTHORISED',
                'RLEASE AUTHORISED','REELASED','RELELASE AUTHROISED.',
                'RELAESD AUTHORISED','RELERASED AUTHORISED.',
                'RELEAED AUTHORISED','RESLD AUTHORISED.', 'SATISFIED  AS', 
                ' RELSD ', 'REL AUTHOURISED', ' RLSD ','RLSD.','RLSD','RLSD BASED ON','.RLSD BASED ON',
                'RELEASED AUTHORISED.','REL AUTHORISED', 'REL AUTHORISED.', '.REL. AUTHORISED.', 
                'RELEASED.','RELEASED!','RELEASED','RELEASE AUTHORIZED','RELEASE AUTHOURIZE.','REL ANTHO',
                'RELEASE BASED ON ABOVE EXAMINATION','RELEASE AUTHOURIZED.','RELEAASED BASE ON','.RELSD',
                ',RSLD','A.I.RSLD','RELEASD AUTHORISED','REALESD','AUTHORISED.','RELSD','RELAESE BASE',
                'AUTHORISED','RELELEASED','RELEASDED','RELESED','RELESE  AUTHORISED.','SATISFIED AS DECLARED.',
                '.RELEASSE AUTHORISED','.REL.ATHORISED.','.REL.AUTHORISED.','RELSED BASED ON EXAMINATION',
                '. RELEAS AUTHORIZED', '. RELASE',',  SATISFIED AS ENTERRED','RELS BASED ON EXM ',
                'REL THUTHORISED.','. AUTHORISED','.RELEASE  AUTHORISED','RELEASE RECOMMENDED','RELSED BASED ON',
                'REEASED AUTHORISED.','RELLEASED.','RLSED',
                '.REL AUTHORSED','RLSED','RELEASD AUTHORISED.','RELEAED','RELEAS RECOMMENDED',
                'RLASD','RELELASE AUTHORISED.','.RLESD','CLEARED','CLEARED BY',
                'RELEAASE AUTHORIZED','RELEASE EFFECTED','.RELEASED BY','RELSD BY']

        zeros= ['AND PAID','UPLIFTED','COMFIRMED','CONFRMED','CONFIRMED','PAYMENT CONFIRMED:',',ISSUED AND PAID',
                '.ISSUED AND PAID','ISSUED AND PAID','PAID','APAID','PAID VIA RECEIPT','ISSUED AND PAID',
                'CONFIRM PAID','CONFIRMED.','DN CONFIRMED PAID','.ISSUED AND PAID','PAYMENT CONFIRMED.',
                'AND  APPROVED','DN CONFIRMED PAID','AND APPROVED',',ISSUED AND PAID',
                'BASED ON THE DN RAISED BY Q & A AND PAID','D/N PAID','DN PAID',
                'REFER TO Q/A FOR FUTHER ACTION','REOUTED','SGD REFERS TO Q&A','REFFER TO Q&A.','BACK TO RED',
                'FRAUDULENT','ITEMS NOT DECLARED','REFER TO Q&A','VALUATION REFERS','REFFER TO Q&A','RED','ENTRY MODIFICATION',
                'MODIFY','TRANSFER OF SHIPMENT','REVENUE REFERS','REFR TO Q&A','REFERED TO','REF TO VALUATION',
                'RE-WRITE',' RE-S ',' FYNA ','TO CPC','UPDATE',' REFERS TO ','. REFER SGD TO APM.','REFERD TO VALUATION',
                ' TFNA ','ATTENTION','LOCAT','CHENGE','REFERS TO Q&A','VALUATION REFERS.','REASIGN TO EXAMINER',
                'CORRECT','FORWARDED','DN APPROVED','CONFIRM PAYMENT BEFORE RELEASE','STRIKE FORCE REFER',
                'MISTAKE','OMITTED','CHANGE TO','NOT DECLARED','REFERED TO CPC','REFD TO VALUATION','RELOC LABEL',
                'REASIGNED','PLS CONFIRM','LOCATION CHANGE','CPC REFERS','REFERED TO VALUATION.','REFD TO SDV VALUATION',
                'REASSIGNMENT',' RE-R ','REFERS FOR FNA','SGD REFERS','VALUATIONN REFERS','AS PPROVED BY CA',
                'IF SATISFIED',' RFD ','NOT SATISFIED','FRAUD DETECTED','.VALUATION REFERS,','RFED FOR DOC CHECK',
                'INPUT EXAM REPORT','PLEASE CHECK','VALUATION/CPC REFERS','TO Q&A','REFFERED TO Q&A/',
                'WRONGLY','WRONG','CANCELLATION','CANCELLED','VALUATION ABOVE REFERS','Q & A REFERS.',
                'CORRECTION','LOCATION CODE CHANGE','VALUATION REFERS.','REFD. TO SDV VALUATION',
                'ACCIDENTED','CHANGE EFFECTED PLEASE','RE-ROUTE','VALUATION REFERS','DC VALUATION, ABOVE',
                'INPUTED WRONGLY','CROSS CHECK','ROUNTED','ROUTED','CPC REFERED.','VALUATION,ABOVE REFERS',
                'BACK TO BULK','SEVCTION','P.E.M','CPC REFER.','SUSPECTED','SONCLEARANCE REQUIRE',
                'CHANGE LOCATION','CHANGE OF LOCATION','OMMITTED','BACK TO RED.','SHORTLANDED','REFRD TO Q & A',
                ' F N A','RE-RETOUTED','TO SCANNING','PLEASE CONFIRM','REASSIGNED TO','REFFERED TO VALUATION',
                'ADDITIONAL  PAYMENT','REQUESTED','.ISSUED','AMMEND','SGD REFER TO VALUATION','REF, TO VAL',
                'ADDITIONAL PAYMENT','MODIFOED TO PAY','REFD TO  VALUATION','DECLARED. Q & A REFERS.',
                'DOES NOT COVER','necrssary action','MODIFIED TO PAY','SGD REFER TO VALUATION.',
                '/ODIFIED TO PAY','PROCEED','ODIFIED TO PAY','REFERS FOR FURTHER ACTION','REFER TO VALUATOIN,',
                'necessary action','RE LOCATION','ALERT!','REFERS FOR FURTHER ACTION.','REFER TO VALUATOIN',
                'ALERT','DN BASED ON UNDER PAYMENT','MODIFUIED TO PAY','REFFERS TO VALUATION',
                'RELOCATION','MODIFIED','YOU MAY PROCEED','REFERS TO VALFFA','Q & A REFERS.',
                'DN BASED ON DECLARATION.','APPROVED','relocation','REFER TO CLASSIC VAL',
                'RFEERS','WRONGLY INPUTED',' COMFIRM ','RELOCATION','VALUARION REFERS','REFFERED TO Q7A/',
                'RE ROUTED',',YOU MAY PROCEED',' RED ','MODIFICATION DONE','CPC (Q7A) REFERS',
                'NECESSARY ACTION','DN RAISED,','AMENDMENT','FORWARDED TO CPC', 'REFEERED TO VALUATION',
                'YOU MAY WISH TO TAKE FURTHER ACTION','FURTHER ACTION','DC VALUATION','CAS APPREOVED BY CAC',
                'REFERS TO QUERY FOR RE-LOCATION.','REFERS TO QUERY','VALUATION REFER.',
                'REROUTED TO TERMINAL','RE ALLOCATED','UNDECLEARED','FWDED TO','DC VALUATION, ABOVE',
                'RE-ROUTED TO RED','DN  RAISED,','FOLLOW UP',' FOLLOW UP','Q&A REFERS.','REFRD T0 VALUATIO',
                'POST ENTRY MODIFICATION DONE TO CHANGE','FOLLOW UP ','FWDED TO VALUATION','EXAMINATION NOT CONDUCTED.',
                'RE ROUTED TO TERMINAL','REFERS TO VAL','REFERS TO ','VALUATION REFERS','REFERES TO Q & A',
                'RE-ROUTED TO YELLOW','RE ROUTE BACK TO BLUE LANE','REFER TO VALUATION','REFD TO',
                'REFER TO VALUATION.','YOU MAY WISH TO PROCEED ACCORDINGLY','VAPPRPVED BY CAC','REFD',
                'YOU MAY WISH TO CONTINUE','RECEIPT NOT','MODIFIED TO PAY','UNDECLARED.Q & A REFERS',
                'VAT PAYMENT','REASSIGNED.','AGAINST','RE ASSIGNED','RELEASD','PENDING PHYSICAL EXAMINATION',
                'RE-CONFIRM','RE ASSIGNED','YOU MAUY WISH TO PROCEED','RECONFIRM','Q&A REFERS.','CPC  REFERS',
                ' PAY ','RE ASSIGNED TO EXAMINER','. REFERED TO EXAMINER.','PENDING PHYSICAL EXAMINATION.',
                'REFERRED TO VALUATION','FOR FOLLOW UP','REFERED TO VALUATION,','REF.TO VALUATION','REF.TO VALUATION.',
                ',YOU MAY WISH TO TAKE FURTHER NECESSARY ACTION AFTER PAYMENT CONFIRMATION.','REFD TO   DOC CHECKS',
                'MODIFUICATION DONE','RE ASSIGNED','REFERED TO VALUATION','YOU MAY TAKE','REFERS.',
                '. RE-ROUTED TO TERMINAL.','.RE-ROUTED TO TERMINAL.','O/C VALUATION','REFD  TO  Q   &  A',
                'RE-ROUTED TO TERMINAL.','YOU MAY WISH',' QUERY ','INPUT VERIFICATION REPORT',
                'FOR FUTHER ACTION','MODIFICATION','ADDING',"VALUATION UNIT REFER",
                ',REFER','RE-ROUTED TO THE TERMINAL','.REFERRED','EXAMINATION NOT CONDUCTED',
                'RE-ASSIGNED TO YELLOW','.CPC REFERRED.','REFERED TO Q&A',
                'RE-ROUTE TO BLUE LANE','RE-ROUTED','REALLOCATED','ISSUED BY VALUATION',
                '. CPC REFEERED','.REFER ','. REFER','INSPECTION','FORCE REFER',
                'CODE AMENDED','ADDITIONAL ITEM CREATED TO PAY','REFFERED TO VALUATION',
                'WRONG CLASSIFICATION',',YOU MAY WISH TO TAKE','REF ;TO VALUATION',
                'CPC REFFERED','REFERRED TO CPC','RE-ASSIGN TO','VALUATION, REFERS',
                'D/N RAISED','YOU MAY PROCEED ACCORDINGLY.','REFER TO SDV VALUATION',
                'RAISED','.RELEASE AUHTORISED',' D/N RAISED ',',D/N RAISED',
                'YOU MAY WISH TO PROCEED AFTER PAYMENT CONFIRMATION.','VALUATION, REFERS.',
                'POST MODIFICATION','RE-ALLOCATED','REFER TO SDV VALUATION.','VAL RFERD',
                'REFERRE', 'YOU MAY PROCEED AFTER CONFIRMATION OF PAYMENT','DC REVENUE REFER',
                ' REFER ','NOT BEEN PAID','REFFERRED','VALAUTION REFERS.','VALAUTION REFERS',
                'YOU MAY WISH TO PROCEED','VALUATION AND Q&A REFERS.','VALUATION AND Q&A REFERS','CPC (Q&A) REFERS']

        def checker(x,each):
            try:
                if each.lower() in x.lower():
                    return True
            except:
                return False
            return False
                        
        for each in zeros:
            X.loc[X.INSPECTION_ACT.apply(lambda x: checker(x,each)),'target'] = 2

        for each in ones:
            X.loc[X.INSPECTION_ACT.apply(lambda x: checker(x,each)),'target'] = 0

        return X

#Slipt Item Column into two features
def Split_Datato_Half(X,y,train_ratio=0.8,Stratified=False):
    """
        This Function Utilizes the Split Functions in Sklearn 
        to Split that into Two halves.
    """
    supported = [numpy.ndarray, pandas.core.frame.DataFrame]
    if type(X) not in supported or type(y) not in supported: 
        raise ValueError(f'X is {type(X)} and y is {type(y)}, both values are expected to be either numpy array or a pandas dataframe')

    split_data = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio) if Stratified else ShuffleSplit(n_splits=1, train_size=train_ratio)
    
    #split the data into two halves
    try:
        X,y = X.values, y.values
    except:
        X,y = X,y

    for train_index, test_index in split_data.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return (X_train,y_train), (X_test, y_test)

#Decode HSCODE feature
class Hscode_Decoder(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        Filling Missing Values with Standard Values That
        Represents Missing Values, e.g numpy.nan.
    """
    def __init__(self,look_url='dataset/CET_tariff.xls'):
        self.look_up = pd.read_excel(look_url)[['CET code','ID']]
        self.look_up = self.look_up.sort_values(by='CET code').set_index(self.look_up['CET code']).drop('CET code',axis=1)
        # print(f'checking the look-up {self.look_up.head()}')

    def fit(self,X):
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset to decode the feature from the lookup table
        """
            
        #Replace Missing Value With Recognized Missing Value
        # X[self.find_in] = X[self.find_in].replace(self.find,self.with_)
        X.loc[X.index,'HSCODE'] = X.HSCODE.apply(lambda x: self.look_up_checker(x))
        return X

    def look_up_checker(self, x):
        try:
            return self.look_up.loc[x].ID
        except:
            return np.nan

#Decode ITMNUM feature
class ItmNum_Decoder(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the itmnum feature.
    """

    def fit(self,X):
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset to decode the feature.
        """
            
        #Replace Missing Value With Recognized Missing Value
        # X[self.find_in] = X[self.find_in].replace(self.find,self.with_)
        X.loc[X.index,'ITM_NUM'] = X.ITM_NUM.apply(lambda x: self.look_up_checker(x))
        return X

    def look_up_checker(self, x):
        try:
            x1,x2 = x.split('/')
            return int(x1)/int(x2)
        except:
            return np.nan

if __name__ == "__main__":
    start = time.time()

    #Because of how large the dataset is, this is a 
    #series of flags that helps fit the data to
    #different methods before transforming during traning
    
    save_dataset = True
    label = True
    encode_strings = False
    encode_labels = False
    encode_hscode = False
    encode_missing_values = False
    drop_hscode_na = False
    replace_digits = False
    encode_classification_fea = False
    encode_itmnum = False
    swap_hsdesc_hscode = False
    split_data = False
    fill_missing = False
    standardize = False
    
    #path to dataset
    # dataset_url = 'dataset/new_dataset2.csv'
    dataset_url = 'dataset/new_data4.csv'

    #all features
    # all_features=[  'IMPORTER_NAME', 'DECLARANT_NAME','OFFICE_NAME', 'CTY_ORIGIN', 'MODE_OF_TRANSPORT', 
    #                 'HSCODE', 'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS',
    #                 'NET_MASS', 'ITM_PRICE', 'STATISTICAL_VALUE', 
    #                 'TOTAL_TAX', 'INVOICE_AMT', 'INSPECTION_ACT','target'
    #             ]
    all_features=[  'IMPORTER_NAME', 'DECLARANT_NAME', 'CTY_ORIGIN', 'MODE_OF_TRANSPORT', 
                    'HSCODE', 'HS_DESC', 'ITM_NUM', 'QTY','GROSS_MASS',
                    'NET_MASS', 'ITM_PRICE', 'STATISTICAL_VALUE', 
                    'TOTAL_TAX', 'INVOICE_AMT', 'INSPECTION_ACT',
                ]
    #target
    target = ['target']
    data = pd.read_csv(dataset_url)[all_features]
    
    #remove degit from dataset
    if replace_digits:
        features = ['INSPECTION_ACT', 'IMPORTER_NAME', 'DECLARANT_NAME','CTY_ORIGIN', 'MODE_OF_TRANSPORT']
        data[features].replace("\d"," ",regex=True)

    #label dataset
    if label:
        data = Labeller().fit_transform(data)

    if encode_strings:
        #Vectorize 'INSPECTION_ACT' 'IMPORTER_NAME', 'HS_DESC','DECLARANT_NAME','OFFICE_NAME' features
        features = ['INSPECTION_ACT', 'IMPORTER_NAME', 'HS_DESC','DECLARANT_NAME','OFFICE_NAME']

        to_find = [None,'None','none','Null','null',np.nan]
        data = Fill_Empty_Spaces(all_features=data.columns, find=to_find, find_in=features,with_=' ').fit_transform(data)
        print('wee dee 1')
        
        strng_enc = String_Encoder(all_features=data.columns, to_encode=features)
        print('wee dee 2')
        strng_enc.fit(data)
        print('wee dee 3')
        dump(strng_enc, 'string_encoder.joblib')
        print('wee dee 4')

    if encode_hscode:
        #vectorize 'HS_DESC' feature
        data = Hscode_Decoder().fit_transform(data)
        print('wee dee 5')

    if drop_hscode_na:
        #vectorize 'HScode' feature
        data.drop(data['HSCODE'][data['HSCODE'].isna()].index, inplace=True)

    if swap_hsdesc_hscode:
        print('hscode')
        data.loc[data.index,'HS_DESC'] = data.HSCODE
        data.drop('HSCODE', axis=1, inplace=True)

    if encode_classification_fea:
        #vectorize 'MODE_OF_TRANSPORT', 'CTY_ORIGIN' features
        features = ['CTY_ORIGIN', 'MODE_OF_TRANSPORT',]
        print('here 1')
        data = Fill_Empty_Spaces(all_features=data.columns, find_in=features,with_='Mode unknown').fit_transform(data)
        print('here 2')
        enc_fea_lab = Encode_Feature_Label(all_features=data.columns, to_encode=features)
        print('here 3')
        enc_fea_lab.fit(data)
        print('here 4')
        dump(enc_fea_lab, 'label_fea_encoder.joblib')

    if encode_itmnum:
        #Encode ITM_NUM
        features = ['ITM_NUM']
        print('here 1')
        data = Fill_Empty_Spaces(all_features=data.columns, find_in=features,with_=np.nan).fit_transform(data)
        data = ItmNum_Decoder().fit_transform(data)
    
    if split_data:
        #Split dataset into train val and test
        y = data[['target']]
        data.drop('target',axis=1,inplace=True)
        X,y,X_rem,y_rem = Split_Datato_Half(X=data, y=y,train_ratio=14,Stratified=True)
        X_train_y_train,X_rem,y_rem = Split_Datato_Half(X=X_rem, y=y_rem,train_ratio=7.41848,Stratified=True)
        X_test_y_test,X_val,y_val = Split_Datato_Half(X=X_rem, y=y_rem)
        pass

    if fill_missing:
        #Fill in missing Values
        pass
    if standardize:
        #Standardize dataset
        pass

    if save_dataset:
        data.to_csv('dataset/new_data4.csv',index=False,)
    print(f'size --> {data.shape}')
    print(f'program executed for {(time.time()-start)/60} mins')
