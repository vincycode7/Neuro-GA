from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit


import pandas as pd
import time
# Drop duplicates
class drop_similars(BaseEstimator, TransformerMixin):
    def __init__(self,to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        """ This is a class to concat multiple similar pandas dataframe together
            by column.
        
        """
        
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X.drop_duplicates()

# Class To Replace Some Characters
class replace_values(BaseEstimator, TransformerMixin):
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
class concat_dsets(BaseEstimator, TransformerMixin):
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
class all_features(BaseEstimator, TransformerMixin):
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

#Slipt Item Column into two features

#Encode categorical features

#Encode numerical features

#Encode String-like features

#Fill missing values

#Label Instances
class labeller(BaseEstimator, TransformerMixin):
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
        assert X.columns == self.features
        return self

    def transform(self,X,y=None):
        X['target'] = 1
        return self.label_datatarget(X)

    @staticmethod
    def label_datatarget(X=None):
        """ function used to assign label value to data, 1 for compliant 
            and 0 for non compliant 
        """

        ones= [ 'RELEASE AUTHORISED',' RELEASE AUTHORISED.',',RELEASE AUTHORISED', 'RELEASE AUTHORISED.','RELSD BY',' REL ','EXD BY','COMFIRMED',
                'RLEASAE',' LEASED ','RELEASDE BASED ON','CORRECTLY CLASSIFIED',' EXED ','TRANSIT TO',
                'FREE TRADE','VERIFIED','BEING ESCORTED','RE- LEASED','AUTH',
                'ON TRANSIT','ON TRANSFER',
                'TAGGED AS ARRIVED','RELSD  BASED',
                ' IN COMPLIANCE','RLS AUTHORISED',
                'EXMND','CONFRMED','CONFIRMED',
                'EXMD BY','RELASED','RELAESED','REALESE','RELEASAED',',ISSUED AND PAID',
                '.ISSUED AND PAID','ISSUED AND PAID','RELAESED','PAID','APAID','RELAS',
                ' RELD ','PAYMENT CONFIRMED:',' RLS ','AUTHORIZED','AUTHORISE.',
                'PAID VIA RECEIPT','RLEASED','ISSUED AND PAID','CONFIRM PAID',
                'RLEASED.','.RLS AUTHORISED','RLEASE RECOMMENDED','REL AUTHO',
                'RLS AUTHORISED','ELEASE AUTHORISED','RLS AUTHORISED.','RELEASA AUTHORISED',
                'RLEASE AUTHORISED','CONFIRMED.','REELASED','RELELASE AUTHROISED.','CLEARED','CLEARED BY',
                'RELAESD AUTHORISED','DN CONFIRMED PAID','.ISSUED AND PAID','RELERASED AUTHORISED.',
                'PAYMENT CONFIRMED.','RELEAED AUTHORISED','RESLD AUTHORISED.', 'SATISFIED  AS', 
                ' ENTERED ', ' RELSD ', 'REL AUTHOURISED','ON TRANSIRE',' RELS ', ' RLSD ',
                'RELEASED AUTHORISED.','REL AUTHORISED', 'REL AUTHORISED.', '.REL. AUTHORISED.', 
                'RELEASED.','RELEASED!','RELEASED','RELEASE AUTHORIZED','RELEASE AUTHOURIZE.',
                'RELEASE BASED ON ABOVE EXAMINATION','RELEASE AUTHOURIZED.',
                '. RELEASE',',RSLD','A.I.RSLD','RELEASD AUTHORISED','REALESD','AUTHORISED.',
                'AUTHORISED','RELELEASED','RELEASDED','RELESED','RELESE  AUTHORISED.','SATISFIED AS DECLARED.',
                '.RELEASSE AUTHORISED','.REL.ATHORISED.','.REL.AUTHORISED.','AND  APPROVED','RELSED BASED ON EXAMINATION',
                '. RELEAS AUTHORIZED','DN CONFIRMED PAID', '. RELASE',',  SATISFIED AS ENTERRED','RELS BASED ON EXM ',
                'REL THUTHORISED.','. AUTHORISED','.RELEASE  AUTHORISED','RELEASE RECOMMENDED','RELSED BASED ON',
                'REEASED AUTHORISED.','RELLEASED.','RLSED','ENTRY MODIFICATION','AND APPROVED',',ISSUED AND PAID',
                '.REL AUTHORSED','RLSED','.REL ','RELEASD AUTHORISED.','ARRIVED','RELEASE','AND PAID',
                'RLASD','RELELASE AUTHORISED.','.RLESD','REL;EASE',' RLEASE ','BASED ON THE DN RAISED BY Q & A AND PAID',
                'RELEAASE AUTHORIZED','UPLIFTED','D/N PAID','DN PAID','RELEASE EFFECTED','.RELEASED BY','RELSD BY']

        zeros= ['REFER TO Q/A FOR FUTHER ACTION','REOUTED','SGD REFERS TO Q&A','REFFER TO Q&A.',
                'FRAUDULENT','ITEMS NOT DECLARED','REFER TO Q&A','VALUATION REFERS','REFFER TO Q&A',
                'MODIFY','TRANSFER OF SHIPMENT','REVENUE REFERS','REFR TO Q&A','REFERED TO',
                'RE-WRITE',' RE-S ',' FYNA ','TO CPC','UPDATE',' REFERS TO ','. REFER SGD TO APM.',
                ' TFNA ','ATTENTION','LOCAT','CHENGE','REFERS TO Q&A','VALUATION REFERS.',
                'CORRECT','FORWARDED','DN APPROVED','CONFIRM PAYMENT BEFORE RELEASE',
                'MISTAKE','OMITTED','CHANGE TO','NOT DECLARED','REFERED TO CPC','REFD TO VALUATION',
                'REASIGNED','PLS CONFIRM','LOCATION CHANGE','CPC REFERS','REFERED TO VALUATION.',
                'REASSIGNMENT',' RE-R ','REFERS FOR FNA','SGD REFERS','VALUATIONN REFERS',
                'IF SATISFIED',' RFD ','NOT SATISFIED','FRAUD DETECTED','.VALUATION REFERS,',
                'INPUT EXAM REPORT','PLEASE CHECK','VALUATION/CPC REFERS','TO Q&A','REFFERED TO Q&A/',
                'WRONGLY','WRONG','CANCELLATION','CANCELLED','VALUATION ABOVE REFERS',
                'CORRECTION','LOCATION CODE CHANGE','VALUATION REFERS.','REFD. TO SDV VALUATION',
                'ACCIDENTED','CHANGE EFFECTED PLEASE','RE-ROUTE','VALUATION REFERS',
                'INPUTED WRONGLY','CROSS CHECK','ROUNTED','ROUTED','CPC REFERED.',
                'BACK TO BULK','SEVCTION','P.E.M','CPC REFER.','SUSPECTED','SONCLEARANCE REQUIRE',
                'CHANGE LOCATION','CHANGE OF LOCATION','OMMITTED','BACK TO RED.',
                ' F N A','RE-RETOUTED','TO SCANNING','PLEASE CONFIRM','REASSIGNED TO',
                'ADDITIONAL  PAYMENT','REQUESTED','.ISSUED','AMMEND',
                'ADDITIONAL PAYMENT','MODIFOED TO PAY','REFD TO  VALUATION',
                'DOES NOT COVER','necrssary action','MODIFIED TO PAY',
                '/ODIFIED TO PAY','PROCEED','ODIFIED TO PAY',
                'necessary action','RE LOCATION','ALERT!',
                'ALERT','DN BASED ON UNDER PAYMENT','MODIFUIED TO PAY',
                'RELOCATION','MODIFIED','YOU MAY PROCEED','REFERS TO VALFFA',
                'DN BASED ON DECLARATION.','APPROVED','relocation',
                'RFEERS','WRONGLY INPUTED',' COMFIRM ','RELOCATION',
                'RE ROUTED',',YOU MAY PROCEED',' RED ','MODIFICATION DONE',
                'NECESSARY ACTION','DN RAISED,','AMENDMENT','FORWARDED TO CPC', 
                'YOU MAY WISH TO TAKE FURTHER ACTION','FURTHER ACTION',
                'REFERS TO QUERY FOR RE-LOCATION.','REFERS TO QUERY',
                'REROUTED TO TERMINAL','RE ALLOCATED','UNDECLEARED','FWDED TO',
                'RE-ROUTED TO RED','DN  RAISED,','FOLLOW UP',' FOLLOW UP',
                'POST ENTRY MODIFICATION DONE TO CHANGE','FOLLOW UP ','FWDED TO VALUATION',
                'RE ROUTED TO TERMINAL','REFERS TO VAL','REFERS TO ','VALUATION REFERS',
                'RE-ROUTED TO YELLOW','RE ROUTE BACK TO BLUE LANE','REFER TO VALUATION',
                'REFER TO VALUATION.','YOU MAY WISH TO PROCEED ACCORDINGLY',
                'YOU MAY WISH TO CONTINUE','RECEIPT NOT','MODIFIED TO PAY',
                'VAT PAYMENT','REASSIGNED.','AGAINST','RE ASSIGNED','RELEASD',
                'RE-CONFIRM','RE ASSIGNED','YOU MAUY WISH TO PROCEED',
                ' PAY ','RE ASSIGNED TO EXAMINER','. REFERED TO EXAMINER.',
                'REFERRED TO VALUATION','FOR FOLLOW UP','REFERED TO VALUATION,'
                ',YOU MAY WISH TO TAKE FURTHER NECESSARY ACTION AFTER PAYMENT CONFIRMATION.',
                'MODIFUICATION DONE','RE ASSIGNED','REFERED TO VALUATION',
                '. RE-ROUTED TO TERMINAL.','.RE-ROUTED TO TERMINAL.',
                'RE-ROUTED TO TERMINAL.','YOU MAY WISH',' QUERY ',
                'FOR FUTHER ACTION','MODIFICATION','ADDING',"VALUATION UNIT REFER",
                ',REFER','RE-ROUTED TO THE TERMINAL','.REFERRED',
                'RE-ASSIGNED TO YELLOW','.CPC REFERRED.','REFERED TO Q&A',
                'RE-ROUTE TO BLUE LANE','RE-ROUTED','REALLOCATED',
                '. CPC REFEERED','.REFER ','. REFER','INSPECTION',
                'CODE AMENDED','ADDITIONAL ITEM CREATED TO PAY',
                'WRONG CLASSIFICATION',',YOU MAY WISH TO TAKE',
                'CPC REFFERED','REFERRED TO CPC','RE-ASSIGN TO',
                'D/N RAISED','YOU MAY PROCEED ACCORDINGLY.',
                'RAISED','.RELEASE AUHTORISED',' D/N RAISED ',',D/N RAISED',
                'YOU MAY WISH TO PROCEED AFTER PAYMENT CONFIRMATION.',
                'POST MODIFICATION','RE-ALLOCATED',
                'REFERRE', 'YOU MAY PROCEED AFTER CONFIRMATION OF PAYMENT',
                ' REFER ','NOT BEEN PAID','REFFERRED',
                'YOU MAY WISH TO PROCEED']

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

if __name__ == "__main__":
    start = time.time()

    label = True
    save = True
    dataset_url = 'dataset/new_dataset2.csv'
    # dataset_url = 'dataset/new_data.csv'
    data = pd.read_csv(dataset_url)

    #label dataset
    if label:
        data = labeller().transform(data)
    if save:
        data.to_csv('dataset/new_data4.csv',index=False,)

    #Vectorize 'INSPECTION_ACT' feature

    #vectorize 'HS_DESC' feature

    #Label Encode 'IMPORTER_NAME'

    #Label Encode 'DECLARANT_NAME'

    #Label Encode 'CTY_ORIGIN'

    #Label Encode 'OFFICE_COD'

    #Label Encode 'OFFICE_NAME' 

    #Label Encide 'MODE_OF_TRANSPORT'

    #Label Encode 'HSCODE'

    #Fill in missing Values

    #Standardize dataset

    print(f'size --> {data.shape}')
    print(f'program executed for {(time.time()-start)/60} mins')
