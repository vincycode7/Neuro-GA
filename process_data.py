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

        ones= [ 'RELEASE AUTHORISED',' RELEASE AUTHORISED.',',RELEASE AUTHORISED', 'RELEASE AUTHORISED.','RELSD BY',' REL ','COMFIRMED',
                'RLEASAE',' LEASED ','RELEASDE BASED ON','CORRECTLY CLASSIFIED',
                'VERIFIED','RE- LEASED','AUTH',
                'RELSD  BASED',
                ' IN COMPLIANCE','RLS AUTHORISED',
                'CONFRMED','CONFIRMED','RLDS BY',
                'RELASED','RELAESED','REALESE','RELEASAED',',ISSUED AND PAID',
                '.ISSUED AND PAID','ISSUED AND PAID','RELAESED','PAID','APAID','RELAS',
                ' RELD ','PAYMENT CONFIRMED:',' RLS ','AUTHORIZED','AUTHORISE.',
                'PAID VIA RECEIPT','RLEASED','ISSUED AND PAID','CONFIRM PAID','RELS BSD ON',
                'RLEASED.','.RLS AUTHORISED','RLEASE RECOMMENDED','REL AUTHO','RELSD BASED ON EXM.',
                'RLS AUTHORISED','ELEASE AUTHORISED','RLS AUTHORISED.','RELEASA AUTHORISED',
                'RLEASE AUTHORISED','CONFIRMED.','REELASED','RELELASE AUTHROISED.','CLEARED','CLEARED BY',
                'RELAESD AUTHORISED','DN CONFIRMED PAID','.ISSUED AND PAID','RELERASED AUTHORISED.',
                'PAYMENT CONFIRMED.','RELEAED AUTHORISED','RESLD AUTHORISED.', 'SATISFIED  AS', 
                ' RELSD ', 'REL AUTHOURISED', ' RLSD ','RLSD.','RLSD','RLSD BASED ON','.RLSD BASED ON',
                'RELEASED AUTHORISED.','REL AUTHORISED', 'REL AUTHORISED.', '.REL. AUTHORISED.', 
                'RELEASED.','RELEASED!','RELEASED','RELEASE AUTHORIZED','RELEASE AUTHOURIZE.','REL ANTHO',
                'RELEASE BASED ON ABOVE EXAMINATION','RELEASE AUTHOURIZED.','RELEAASED BASE ON','.RELSD',
                '. RELEASE',',RSLD','A.I.RSLD','RELEASD AUTHORISED','REALESD','AUTHORISED.','RELSD','RELAESE BASE',
                'AUTHORISED','RELELEASED','RELEASDED','RELESED','RELESE  AUTHORISED.','SATISFIED AS DECLARED.',
                '.RELEASSE AUTHORISED','.REL.ATHORISED.','.REL.AUTHORISED.','AND  APPROVED','RELSED BASED ON EXAMINATION',
                '. RELEAS AUTHORIZED','DN CONFIRMED PAID', '. RELASE',',  SATISFIED AS ENTERRED','RELS BASED ON EXM ',
                'REL THUTHORISED.','. AUTHORISED','.RELEASE  AUTHORISED','RELEASE RECOMMENDED','RELSED BASED ON',
                'REEASED AUTHORISED.','RELLEASED.','RLSED','AND APPROVED',',ISSUED AND PAID',
                '.REL AUTHORSED','RLSED','RELEASD AUTHORISED.','RELEASE','AND PAID','RELEAED','RELEAS RECOMMENDED',
                'RLASD','RELELASE AUTHORISED.','.RLESD','REL;EASE',' RLEASE ','BASED ON THE DN RAISED BY Q & A AND PAID',
                'RELEAASE AUTHORIZED','UPLIFTED','D/N PAID','DN PAID','RELEASE EFFECTED','.RELEASED BY','RELSD BY']

        zeros= ['REFER TO Q/A FOR FUTHER ACTION','REOUTED','SGD REFERS TO Q&A','REFFER TO Q&A.','BACK TO RED',
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
