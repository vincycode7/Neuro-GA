from process_data import *
import pandas as pd

#Test 1
dataset1 = pd.DataFrame([   'FREE TRADE','VERIFIED',
                            'RE-ROUTE','INPUTED WRONGLY',
                            'TAGGED AS ARRIVED','EXAMINATION',
                            'EXAMS BY',' IN COMPLIANCE',
                            'MODIFIED','YOU MAY PROCEED',
                            'DN BASED ON DECLARATION.',
                            'Sleeping'
                        ], columns=['INSPECTION_ACT'])

lst =   [['FREE TRADE',1],['VERIFIED',1],
        ['RE-ROUTE',0],['INPUTED WRONGLY',0],
        ['TAGGED AS ARRIVED',1],['EXAMINATION',1],
        ['EXAMS BY',1],[' IN COMPLIANCE',1],
        ['MODIFIED',0],['YOU MAY PROCEED',0],
        ['DN BASED ON DECLARATION.',0],
        ['Sleeping',1]]
ans1 =    pd.DataFrame(lst, columns=['INSPECTION_ACT','target'])

#Test2 

dataset2 = pd.DataFrame([   'FREE TRADE','VERIFIED',
                            'RE-ROUTE','INPUTED WRONGLY',
                            '',
                            'TAGGED AS ARRIVED','EXAMINATION',
                            ' ',
                            'EXAMS BY',' IN COMPLIANCE',
                            '....',
                            'MODIFIED','YOU MAY PROCEED',
                            'DN BASED ON DECLARATION.',
                            '   ',
                            'Sleeping'
                            '.......'
                        ], columns=['INSPECTION_ACT'])

ans2 = ['FREE', 'TRADE', 'VERIFIED', 'RE-ROUTE', 'INPUTED', 'WRONGLY', 
        'TAGGED','AS', 'ARRIVED', 'EXAMINATION', 'EXAMS', 'BY', 'IN', 
        'COMPLIANCE', 'MODIFIED', 'YOU', 'MAY', 'PROCEED', 'DN', 'BASED', 
        'ON', 'DECLARATION', 'Sleeping']

class label_data:
    def __call__(self,X):
        return Labeller().transform(X) == ans1

class encode_strings:
    def __call__(self,X):
        return String_Encoder().transform(X) == ans2
        
def main():
    print(f'result {label_data()(dataset1)}')
    print(f'result {encode_strings()(dataset2)}')
if __name__ == "__main__":
    main()