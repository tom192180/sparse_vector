import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, AutoModel
import json
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import argparse
from scipy.optimize import linear_sum_assignment
import numpy as np
import re
from sklearn.metrics import cluster
from tqdm import tqdm
import csv
import os
import torch.nn.functional as F
from collections import defaultdict
from sklearn.preprocessing import normalize
from collections import Counter
import hdbscan
from sklearn.metrics import pairwise_distances_argmin_min

class Evaluation:
    def __init__(self,file_path, seed_value, m, method):
        self.seed_value = seed_value
        self.m = m # m  = budget num * partition num
        self.embeddings = None
        self.file_path = file_path
        self.method = method
        self.iter = Evaluation._get_iter_number(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        filename = file_path.split('/')[-1]
        filename_list = filename.split('_')
        dataset_names = ['clinc','clincKIR','banking','bankingKIR','mtop','massive','emo','stackoverflow','stackoverflowKIR', 'arxiv', 'reddit']
        dataset_name_id = [id for id, dataset_name in enumerate(filename_list) if dataset_name in dataset_names][0]
        self.dataset_name =  filename_list[dataset_name_id]
        self.llm_name = filename_list[dataset_name_id+1]
        self.retriever_name = self._get_retriever_name(filename_list[dataset_name_id+2])    

        
        self.test_examples_txt = [item['text'] for item in data]
        self.test_examples_label = [
            item['Ground Truth'].replace('_', ' ') if isinstance(item['Ground Truth'], str) else item['Ground Truth']
            for item in data
        ]

        self.cluster_num = len(set(self.test_examples_label))
 
        file_path_dense_emb = file_path.replace('.json', 'D.npz')
        file_path_sparse_emb = file_path.replace('.json', 'S.npz')

        test_examples_txt_dense_embd = np.load(file_path_dense_emb)
        test_examples_txt_sparse_embd = np.load(file_path_sparse_emb)

        self.dense_emb = test_examples_txt_dense_embd['embeddings']
        self.sparse_emb = test_examples_txt_sparse_embd['embeddings']
        print(type(self.dense_emb), self.dense_emb.dtype)

        print(f'dense embedding shape: {self.dense_emb.shape}; sparse embedding shape: {self.sparse_emb.shape}')


        if 'KIR' in file_path:
            # Because part of embeddings ar from addional data, we do not need them
            dataset_name_KIR_test = self.dataset_name.replace('KIR', '')
            kir_path = f'../ALUP/data/{dataset_name_KIR_test}/test.tsv'
            list_of_uts, _ = Evaluation._read_tsv(filepath = kir_path)
            
            id_map = {txt: i for i, txt in enumerate(self.test_examples_txt)}
            subset_ids = [id_map[ut] for ut in list_of_uts if ut in id_map]
            # Filter everything using these indices
            self.test_examples_txt = [self.test_examples_txt[i] for i in subset_ids]
            self.test_examples_label = [self.test_examples_label[i] for i in subset_ids]
            self.dense_emb = [self.dense_emb[i] for i in subset_ids]
            self.sparse_emb = [self.sparse_emb[i] for i in subset_ids]

        print(f'Ground truth data num: {len(self.test_examples_txt)}; Ground truth unique intent num: {self.cluster_num}')

            

    @staticmethod
    def _read_tsv(filepath):
        list_of_uts = []
        list_of_intents = []

        with open(filepath, 'r', encoding='utf-8') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            next(tsvreader)  # skip header
            for row in tsvreader:
                if not row:  # skip empty rows
                    continue
                list_of_uts.append(row[0].strip())
                list_of_intents.append(row[1].strip())

        return list_of_uts, list_of_intents



    def _get_retriever_name(self,retriever_name):
        if 'e5-large' in retriever_name:
            return 'intfloat/e5-large'
        elif 'instructor' in retriever_name:
            return 'hkunlp/instructor-large'
        elif 'MiniLM' in retriever_name:
            return 'sentence-transformers/all-MiniLM-L6-v2'
        elif 'bert' in retriever_name:
            return 'google-bert/bert-base-uncased'      

    @staticmethod
    def _get_iter_number(file_path):
        match = re.search(r'Iter(\d+)', file_path)
        return match.group(1) if match else 0




    # original version
    def evaluate(self,mode): 
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(self.test_examples_label)
        
        # ----- KMeans -----
        if self.method == 'K':
            for i in range(1,6):
                self.seed_value = i
                kmeans = KMeans(n_clusters=self.cluster_num, random_state = self.seed_value, n_init=10)
                clusters_kmean = kmeans.fit_predict(self.embeddings)
                nmi = normalized_mutual_info_score(label_ids, clusters_kmean)
                acc = Evaluation.clustering_accuracy(label_ids, clusters_kmean)


                self._save_metrics_to_csv(mode, nmi, acc, 'kmean') 
            
        # ----- HDBSCAN -----
        elif self.method == 'H': 
            param_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_score = -1
            best_clusters = None
            best_param = None
            self.embeddings = np.array(self.embeddings, dtype=float)
            for size in param_values:
                hdb = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=1, metric='euclidean')
                clusters_hdb = hdb.fit_predict(self.embeddings)
                clusters_hdb = np.array(clusters_hdb)
                

                # Boolean masks
                noise_mask = clusters_hdb == -1
                non_noise_mask = clusters_hdb != -1  # <- must define this before using in if

                if np.any(noise_mask) and np.any(non_noise_mask): # at least one noise point and at least one cluster
                    unique_clusters = np.unique(clusters_hdb[non_noise_mask]).astype(int)
                    centroids_list = []
                    for c in unique_clusters:
                        cluster_points = self.embeddings[clusters_hdb == c] 
                        centroid = cluster_points.mean(axis=0)
                        centroids_list.append(centroid)
                    centroids = np.array(centroids_list)

                    noise_points = self.embeddings[noise_mask]
                    closest, _ = pairwise_distances_argmin_min(noise_points, centroids)
                    clusters_hdb[noise_mask] = unique_clusters[closest]
                try: # there is only one cluster
                    score = silhouette_score(self.embeddings, clusters_hdb)
                except ValueError:
                    score = -1

                print(f"min_cluster_size={size}, clusters={len(np.unique(clusters_hdb))}, silhouette={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_clusters = clusters_hdb.copy()
                    best_param = size
                
                else:
                    print("No improvement, stopping search.")
                    break  # <- early stopping here
            if best_clusters is None:
                # fallback: everything as one cluster
                best_clusters = np.zeros(len(self.embeddings), dtype=int)
                print("Warning: No valid HDBSCAN clustering found; assigning all points to one cluster.")
                best_param = 0

            best_non_noise_mask = best_clusters != -1
            self.hdbscan_num_clusters = len(np.unique(best_clusters[best_non_noise_mask])) if np.any(best_non_noise_mask) else 0

            self.best_param = best_param
            nmi_hdb = normalized_mutual_info_score(label_ids, best_clusters)
            acc_hdb = Evaluation.clustering_accuracy(label_ids, best_clusters)
            self._save_metrics_to_csv(mode, nmi_hdb, acc_hdb, 'hdbscan')
            print("Best param (min_cluster_size = min_samples):", best_param, "Best Silhouette:", best_score)


    @staticmethod
    def _pad_field(value, width):
        return str(value).ljust(width)


    def _save_metrics_to_csv(self,mode,nmi, acc, clustering):
        # Dictionary of metric results
        word_dir = f'../eval/'
        # Ensure the directory exists
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)        
        llm_name = self.llm_name.split('/')[-1]
        retriever_name = self.retriever_name.split('/')[-1]

        if 'full' in self.file_path:
            share_file_name = f'{self.dataset_name}_{llm_name}_{retriever_name}_fullDim'
        else:
            match = re.search(r'_(\d+)dim', self.file_path)
            if match:
                number = int(match.group(1))
            share_file_name = f'{self.dataset_name}_{llm_name}_{retriever_name}_{number}Dim'


        file_name=f'{word_dir + share_file_name}_{clustering}.csv'


        print(f'Ｔhe data is saved to:{file_name}')
        if clustering == 'kmean':
            metrics = {
                'Mode': mode,
                'Seed': self.seed_value,
                'Iter': self.iter,
                'NMI': f"{nmi * 100:.2f}",
                'ACC': f"{acc * 100:.2f}",
                'Budget': self.m,
            }

            widths = {
                'Mode': 15,
                'Seed': 5,
                'Iter': 5,
                'NMI': 6,
                'ACC': 6,
                'Budget': 8,
            }
        elif clustering == 'hdbscan':
            metrics = {
                'Mode': mode,
                'Seed': self.seed_value,
                'Iter': self.iter,
                'NMI': f"{nmi * 100:.2f}",
                'ACC': f"{acc * 100:.2f}",
                'Budget': self.m,
                'Khat': self.hdbscan_num_clusters,
                'K': self.cluster_num,
                'Par': self.best_param,
            }

            widths = {
                'Mode': 15,
                'Seed': 5,
                'Iter': 5,
                'NMI': 6,
                'ACC': 6,
                'Budget': 8,
                'Khat': 5,
                'K': 5,
                'Par': 5,
            }

        metrics = {Evaluation._pad_field(k, widths[k]): Evaluation._pad_field(v, widths[k]) for k, v in metrics.items()}


        # Write metrics to CSV
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)


    @staticmethod
    def clustering_accuracy(true_labels, cluster_labels):
        contingency_matrix = cluster.contingency_matrix(true_labels, cluster_labels)
        row_ind, col_ind = linear_sum_assignment(contingency_matrix, maximize=True)
        optimal_assignment = contingency_matrix[row_ind, col_ind].sum()
        accuracy = optimal_assignment / len(true_labels)
        return accuracy



parser = argparse.ArgumentParser(description='LLM evaluation')
parser.add_argument('--file_path', type=str)
parser.add_argument('--method',default = 'H', type=str)


 






args = parser.parse_args()
file_path = args.file_path
method = args.method

# Match digits before and after 'sd'
match = re.search(r'(\d+)sd(\d+)', file_path)

if match:
    seed_value = int(match.group(1))
    m = int(match.group(2))

# Reproducible
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  
np.random.seed(seed_value)
random.seed(seed_value) 


 
eval_cluster = Evaluation(file_path = file_path, seed_value = seed_value, m = m, method = method)
test_examples_txt = eval_cluster.test_examples_txt
dense_emb = eval_cluster.dense_emb
sparse_emb = normalize(eval_cluster.sparse_emb, norm='l2', axis=1)



# eval_cluster.embeddings = dense_emb
# eval_cluster.evaluate(mode = 'Dense')

eval_cluster.embeddings = sparse_emb
eval_cluster.evaluate(mode = 'Sparse')













