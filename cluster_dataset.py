import csv
from datasets import load_dataset
import random

class ClusterDataset:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

    def load_data(self):
        """
        Load dataset based on self.dataset_name and return raw (ut, intent) lists.

        Returns:
            list_of_uts (list): List of user texts/questions.
            list_of_intents (list): List of corresponding labels/intents.
        """
        if 'banking' == self.dataset_name:
            ds = load_dataset("legacy-datasets/banking77")
            list_of_uts = [example['text'] for example in ds['test']]
            list_of_intents = [example['label'] for example in ds['test']]
        else:
            elif 'KIR' in self.dataset_name:
                if 'clincKIR' == self.dataset_name:    
                    filepath1 = '../ALUP/data/clinc/test.tsv'
                    filepath2 = '../ALUP/data/clinc/train.tsv'

                elif 'bankingKIR' == self.dataset_name:    
                    filepath1 = '../ALUP/data/banking/test.tsv'
                    filepath2 = '../ALUP/data/banking/train.tsv'

                elif 'stackoverflowKIR' == self.dataset_name: 
                    filepath1 = '../ALUP/data/stackoverflow/test.tsv'
                    filepath2 = '../ALUP/data/stackoverflow/train.tsv'


            elif 'clinc' == self.dataset_name:
                filepath = '../clinc/small.csv'
            elif 'mtop' == self.dataset_name:
                filepath = '../mtop_intent/small.csv'
            elif 'massive' == self.dataset_name:  
                filepath = '../massive_intent/small.csv'
            elif 'emo' == self.dataset_name:  
                filepath = '../go_emotion/small.csv'
            elif 'reddit' == self.dataset_name:  
                filepath = '../reddit/small.csv'
            elif 'arxiv' == self.dataset_name:  
                filepath = '../arxiv_fine/small.csv'
            elif 'stackoverflow' == self.dataset_name:
                filepath = '../ALUP/data/stackoverflow/test.tsv'
            else:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")



            if 'stackoverflow' == self.dataset_name:
                list_of_uts, list_of_intents = ClusterDataset._read_tsv(filepath = filepath)

            elif 'KIR' in self.dataset_name:
                list_of_uts_test, list_of_intents_test = ClusterDataset._read_tsv(filepath = filepath1)
                list_of_uts_train, list_of_intents_train = ClusterDataset._read_tsv(filepath = filepath2)

                # set seed for reproducibility
                random.seed(42)

                combined = list(zip(list_of_uts_train, list_of_intents_train))
                random.shuffle(combined)
                list_of_uts_train, list_of_intents_train = zip(*combined)
                list_of_uts_train, list_of_intents_train = list(list_of_uts_train), list(list_of_intents_train)
                        
                
                
                ratio_train = 0.3
                n = int( ratio_train * len(list_of_uts_train))
                print(f'The ratio used for train split {ratio_train}')
                list_of_uts = list_of_uts_test + list_of_uts_train[:n]
                list_of_intents = list_of_intents_test + list_of_intents_train[:n]
            else:
                list_of_uts = []
                list_of_intents = []
                with open(filepath, 'r', encoding='utf-8') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        if not row:
                            continue
                        list_of_uts.append(row[0].strip())
                        list_of_intents.append(row[1].strip())

        return list_of_uts, list_of_intents



    @staticmethod
    def _read_tsv(filepath):
        list_of_uts = []
        list_of_intents = []

        with open(filepath, 'r', encoding='utf-8') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            next(tsvreader)  # to skip the header
            for row in tsvreader:
                if not row:  # skip empty rows
                    continue
                list_of_uts.append(row[0].strip())
                list_of_intents.append(row[1].strip())

        return list_of_uts, list_of_intents
