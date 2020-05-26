Write-UP For ProJect

[Intro]
This Project Is one that aims to explore genetic algorithm and it's application in optimizing neural network for better performance. In this project the main field the two algorithms will be applied to is the *Custom field*, where goods are been checked for any fraudulent act. The project aims to achieve success in training a model that can successfully Identify fraudulent act in the process of importing goods thereby reducing time allocated to checking goods by narrowing search to goods that really call for concern. 

[The Dataset]
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




[The Algorithm]


                                                    << About the Artificial neural network >>

                                                        <<  About Genetic Algorithm >>

[The Method]

                                                        ** Selection Operator **

                                                        ** Cross-Over Operator **

                                                        ** Mutation Operator **

                                                             ** Elitism **

[What is been optimuzed]

