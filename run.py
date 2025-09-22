import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, AutoModel
import torch
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import csv
import json
import random
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import argparse
import string
from InstructorEmbedding import INSTRUCTOR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datasets import load_dataset
from prompt_template import PromptTemplate
from cluster_dataset import ClusterDataset
from sklearn.cluster import KMeans, AgglomerativeClustering
import re



class InContextLLM:
    
    def __init__(self, 
                 retriever_name,
                 llm_name,
                 dataset_name, 
                 budget_num,
                 partition_num,
                 initial_cluster,
                 initial_dimension,
                 seed_value):
        self.llm_name = llm_name
        self.retriever_name = retriever_name
        self.dataset_name = dataset_name
        self.iter = 0
        self.embeddings = None
        self.initial_dimension = initial_dimension
        self.budget_num = budget_num # number of example given in the prompt
        self.initial_cluster = initial_cluster
        self.num_partitions = partition_num
        self.closest_ut_list = []
        self.llm_output = []
        self.total_unique_pred_label_num = []
        self.sd = seed_value
        print(f'The retriever is: {self.retriever_name}')
        print(f'The llm is: {self.llm_name}')

        # Data Processing

        
        n = 1000000
        
        
        # test data 
    
        self.test_examples_txt, self.test_examples_label = self.load_data()
        self.test_examples_txt, self.test_examples_label = self.test_examples_txt[:n], self.test_examples_label[:n]

        print(f'There are {len(set(self.test_examples_label))} unique intents.')
        print(f'There are {len(self.test_examples_txt)} examples.')

        self.text_to_label = dict()
        self.text_to_label_list = dict()

        # label appended data
        self.example_rep_label_list =  self.test_examples_txt[:] 
        self.text_label_feq = dict() # it will be a dictionary of dictionary
        self.group_label = set()
        self.text_to_sparse_vec = {text:None for text in self.test_examples_txt}
        self.text_to_dense_vec = None
        self.text_to_convergence = {text:0 for text in self.test_examples_txt}
        self.text_to_gd_label = {text:gd for text, gd in zip(self.test_examples_txt,self.test_examples_label)}
        # LLM
        torch_dtype = torch.bfloat16 if 'Qwen' in self.llm_name else torch.float16
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name,
                                                       token= '',
                                                       padding_side="left")
        self.llm  = AutoModelForCausalLM.from_pretrained(self.llm_name,torch_dtype=torch_dtype,
                                                         token= '',
                                                         device_map='auto')
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id 



        # Retriever
        if 'sentence-transformers' in self.retriever_name or 'e5-large' in self.retriever_name or 'bert' in self.retriever_name:
            self.retriever_tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)
            self.retriever = AutoModel.from_pretrained(self.retriever_name, device_map='auto')
        elif 'instructor' in self.retriever_name:
            self.retriever = INSTRUCTOR('../instructor-large', device='cuda')

        
    def load_data(self):
        
        random.seed(self.sd)
        data = ClusterDataset(dataset_name = self.dataset_name)
        list_of_uts, list_of_intents = data.load_data()
        combined = list(zip(list_of_uts, list_of_intents))
        random.shuffle(combined)
        shuffled_uts, shuffled_intents = zip(*combined)

        shuffled_uts = list(shuffled_uts)
        shuffled_intents = list(shuffled_intents)
            



        return shuffled_uts, shuffled_intents
        


     

    def _find_closest_sentences(self, mode):
        self.closest_ut_list.clear()
        num_test = len(self.test_examples_txt)
        
        if mode == 'initial':
            test_embeddings = self.embeddings
            compared_embeddings = self.representative_embeddings
            distance_matrix = pairwise_distances(test_embeddings, compared_embeddings, metric='cosine')  
            compared_text = self.representative_texts


        elif mode == 'iterative':
            sparse_vecs = [self.text_to_sparse_vec[text] for text in self.test_examples_txt]
            dense_vecs = [self.text_to_dense_vec[text] for text in self.test_examples_txt]
            # Convert to arrays if not already
            sparse_array = np.array(sparse_vecs)
            dense_array = np.array(dense_vecs)

            # Concatenate vectors horizontally (feature axis)
            test_embeddings = compared_embeddings = np.concatenate((sparse_array, dense_array), axis=1)
            distance_matrix = pairwise_distances(test_embeddings, compared_embeddings, metric='cosine')  
            np.fill_diagonal(distance_matrix, np.inf)
            compared_text = self.test_examples_txt

          
        np.random.seed(self.sd)
        for i in range(num_test):
            distances_to_train = distance_matrix[i]
            # to get inds of top n * budget_num candidates
            top_indices = np.argsort(distances_to_train)[:self.budget_num * self.num_partitions]

            # shuffle these inds to randomize
            shuffled = np.random.permutation(top_indices)

            # split into n partitions
            partitions = [
                shuffled[j*self.budget_num : (j+1)*self.budget_num]
                for j in range(self.num_partitions)
            ]

            # convert each partition into sentences
            close_sent_lists = [
                [compared_text[j] for j in part] for part in partitions
            ]

            # append tuple which contains n lists
            self.closest_ut_list.append(tuple(close_sent_lists))


    def get_embeddings(self):
        self.retriever.eval()
        
        batch_size = 60 
        all_embeddings = []  
        example_list =  self.test_examples_txt
        for i in tqdm(range(0, len(example_list), batch_size)):
            batch_ut = example_list[i:i + batch_size]
            if 'instructor' in self.retriever_name:

                if 'bank' in self.dataset_name:
                    sentences = [['Represent the bank purpose for retrieval: ',ut] for ut in batch_ut]
                elif 'clinc' in self.dataset_name or 'massive' in self.dataset_name:
                    sentences = [['Represent the sentence for retrieving the purpose: ',ut] for ut in batch_ut]
                elif 'mtop' in self.dataset_name or 'stackoverflow' in self.dataset_name:
                    sentences = [['Represent the sentence for retrieval: ',ut] for ut in batch_ut]
                elif 'emo' in self.dataset_name:
                    sentences = [['Represent an emotion sentence for retrieval: ',ut] for ut in batch_ut]
                elif 'reddit' in self.dataset_name:
                    sentences = [['Represent a reddit community title: ',ut] for ut in batch_ut]
                elif 'arxiv' in self.dataset_name:
                    sentences = [['Represent the science statement for retrieval: ',ut] for ut in batch_ut]

                utterance_embeddings = self.retriever.encode(sentences)
                utterance_embeddings = torch.from_numpy(utterance_embeddings)
 
            else:
                if 'e5' in self.retriever_name:
                    batch_ut = ['query: ' + ut for ut in batch_ut]
                inputs = self.retriever_tokenizer(batch_ut, padding=True, return_tensors="pt")
                inputs = {key: value.to(self.retriever.device) for key, value in inputs.items()}

                attention_mask = inputs['attention_mask']
                with torch.no_grad():
                    # mean pooling
                    utterance_embeddings = self.retriever(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]   
                    pooled = torch.sum(utterance_embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
                    pooled.masked_fill_(torch.isnan(pooled), 0)    
                    utterance_embeddings = pooled

                utterance_embeddings = utterance_embeddings.cpu()
            all_embeddings.append(utterance_embeddings)  
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.float()  
        self.embeddings_raw = all_embeddings
        normalized_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        self.embeddings =  normalized_embeddings.numpy()   
        self.text_to_dense_vec = {text:emb for text, emb in zip(self.test_examples_txt, self.embeddings)}


        del self.retriever  
        torch.cuda.empty_cache()        

        
            
    def _sent_selection_prompt(self, mode):
        """
        The method is to generate the LLM selection prompt for each text, the output will unpack the tuple.   
        """
        input_text = []
        for target, demo_partitions in zip(self.test_examples_txt, self.closest_ut_list):
            if (self.text_to_sparse_vec[target] is None
                or (mode == 'iterative' and self.text_to_convergence[target] == 0)
            ):
                # generate a prompt for each partition, demo_partitions has n partition
                for demo_examples in demo_partitions:
                    prompt_bg = PromptTemplate.render_sent_selection_prompt(
                        dataset_name=self.dataset_name,
                        target=target,
                        demo_examples=demo_examples
                    )
                    input_text.append(prompt_bg)

        return input_text



    
    def group_with_llm(self, mode):
        self.llm_output.clear()

        llm_name, retriever_name = self.llm_name.split('/')[-1], self.retriever_name.split('/')[-1]  
        self.word_dir = f'../output/'
        # Ensure the directory exists
        if not os.path.exists(self.word_dir):
            os.makedirs(self.word_dir)        
        self.share_file_name = f'{self.dataset_name}_{llm_name}_{retriever_name}_{self.sd}sd{int(self.budget_num * self.num_partitions)}_{self.initial_dimension}dim{self.initial_cluster}'
        

        self._find_closest_sentences(mode = mode)
        filename = f'{self.word_dir + self.share_file_name}'
        input_text= self._sent_selection_prompt(mode = mode)    
            
        sep_words = [] # to get prompt length
        input_text_add_tmp = []
        
        for chunk in input_text:
            if 'llama' in self.llm_name or 'Qwen2.5' in self.llm_name:
                sys_instruct = "You are a chatbot that always answers with accurate responses"    
                text_format = [{"role": "system","content": sys_instruct,},{"role": "user", "content": chunk}] 
                text_with_cht_tmp = self.tokenizer.apply_chat_template(text_format, add_generation_prompt=True, tokenize=False)

            elif 'gemma' in self.llm_name: 
                text_format = [{"role": "user", "content": chunk}]
                text_with_cht_tmp = self.tokenizer.apply_chat_template(text_format, add_generation_prompt=True, tokenize=False)

            elif 'Qwen3' in self.llm_name:
                text_format = [{"role": "user", "content": chunk}]
                text_with_cht_tmp = self.tokenizer.apply_chat_template(text_format, add_generation_prompt=True, tokenize=False,enable_thinking=False)
            
            sp = text_with_cht_tmp[-200:]
            input_text_add_tmp.append(text_with_cht_tmp)
            sep_words.append(sp)  
        
        batch_size = 100 
        closest_pair_id = []
        
        for i in range(0, len(input_text_add_tmp), batch_size):
            batch_sep_words_tk = self.tokenizer(sep_words[i:i + batch_size], padding=True,  return_tensors="pt") # to make sure the format is the same as input en-de 
            batch_texts = input_text_add_tmp[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True,  return_tensors="pt")
            inputs = {key: value.to(self.llm.device) for key, value in inputs.items()}

            outputs = self.llm.generate(**inputs,max_new_tokens = 200)    
        
            # Decode generated tokens to text
            batch_generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_sep_words = self.tokenizer.batch_decode(batch_sep_words_tk['input_ids'], skip_special_tokens=True) # to make sure the format is the same as input en-de 


            for pred_text , sep_word in zip(batch_generated_texts, batch_sep_words):
                pred_split = pred_text.split(sep_word)
                pred = pred_split[-1].lower()
                tmp = []
                if 'none' in pred: # consider use second loop
                    tmp = tmp
                else:
                    numbers = [int(num)-1 for num in re.findall(r'\d+', pred)] 
                    for ind in numbers:
                        if ind >= self.budget_num:
                            continue
                        else:
                            tmp.append(ind)

                closest_pair_id.append(tmp) 





        merge_every_n_partitions = []
        for i in range(0, len(closest_pair_id), self.num_partitions):
            parts = [closest_pair_id[i + j] for j in range(self.num_partitions)]
            merge_every_n_partitions.append(tuple(parts))
        closest_pair_id = merge_every_n_partitions

        # Get sparse vector
        if mode == 'initial':
            # only construct none-representative text sparse vector
            filtered_pairs = [(text, ut_list) for text,  ut_list in zip(self.test_examples_txt,self.closest_ut_list) if self.text_to_sparse_vec[text] is None]
            test_examples_txt_remain = [text for text, _ in filtered_pairs]
            closest_ut_list_remain = [ut_list for _, ut_list in filtered_pairs]
            self._build_sparse_vec(closest_pair_id = closest_pair_id,
                                   closest_ut_list_remain = closest_ut_list_remain,
                                   test_examples_txt_remain = test_examples_txt_remain)
            self._extend_dimension()
        
        elif mode == 'iterative':
            # only update sparse vector yet to  convergence
            filtered_pairs = [(text, ut_list) for text,  ut_list in zip(self.test_examples_txt,self.closest_ut_list) if self.text_to_convergence[text] == 0]
            test_examples_txt_remain = [text for text, _ in filtered_pairs]
            closest_ut_list_remain = [ut_list for _, ut_list in filtered_pairs]


            self.iter += 1
            filename += f'Iter{self.iter}'
            self._update_vec(closest_pair_id = closest_pair_id,
                             closest_ut_list_remain = closest_ut_list_remain,
                             test_examples_txt_remain = test_examples_txt_remain)


            
                
        

        self._save_text_embeddings(filepath = filename)




    def _build_sparse_vec(self, closest_pair_id, closest_ut_list_remain, test_examples_txt_remain):
        
        if len(closest_pair_id) != len(closest_ut_list_remain) or len(closest_pair_id) != len(test_examples_txt_remain):
            raise ValueError(f"Length mismatch: closest_pair_id={len(closest_pair_id)}, "
                             f"closest_ut_list_remain={len(closest_ut_list_remain)}, "
                             f"test_examples_txt_remain={len(test_examples_txt_remain)}")
            
        for ind_tuple, ut_tuple, text in zip(closest_pair_id, closest_ut_list_remain, test_examples_txt_remain):
            
            all_vecs = []
            for ind_list, ut_list in zip(ind_tuple, ut_tuple):
                if len(ind_list) > 0:
                    for ind in ind_list:
                        positive_text = ut_list[ind]
                        all_vecs.append(self.text_to_sparse_vec[positive_text])

            if len(all_vecs) > 0:
                embd = np.sum(all_vecs, axis=0) / len(all_vecs)

                # L2 normalization
                norm = np.linalg.norm(embd)
                embd = embd / norm

                self.text_to_sparse_vec[text] = embd


    def _update_vec(self, closest_pair_id, closest_ut_list_remain, test_examples_txt_remain):      
        
        if len(closest_pair_id) != len(closest_ut_list_remain) or len(closest_pair_id) != len(test_examples_txt_remain):
            raise ValueError(f"Length mismatch: closest_pair_id={len(closest_pair_id)}, "
                             f"closest_ut_list_remain={len(closest_ut_list_remain)}, "
                             f"test_examples_txt_remain={len(test_examples_txt_remain)}")        
        
        for ind_tuple, ut_tuple, text in zip(closest_pair_id, closest_ut_list_remain, test_examples_txt_remain):
            old_sparse_embd = self.text_to_sparse_vec[text]
            

            # collect positives across all partitions
            positive_texts = []
            for ind_list, ut_list in zip(ind_tuple, ut_tuple):
                positive_texts.extend([ut_list[i] for i in ind_list])

            if len(positive_texts) > 0:
                sparse_embd = InContextLLM._average_and_normalize(
                    old_sparse_embd, positive_texts, self.text_to_sparse_vec
                )
                
            else:
                sparse_embd = old_sparse_embd
                

            # Check convergence

            sparse_cosine_sim = np.dot(old_sparse_embd, sparse_embd)
            
            if sparse_cosine_sim > 0.99:
                self.text_to_convergence[text] = 1                
                self.text_to_sparse_vec[text] = old_sparse_embd
            else:
                self.text_to_sparse_vec[text] = sparse_embd

        # After update, print global convergence stats:
        total = len(self.text_to_convergence)
        converged = sum(v == 1 for v in self.text_to_convergence.values())
        not_converged = total - converged
        print(f"{self.iter}th iter: {converged} converged, {not_converged} still iterating (out of {total})")
        
        # Log the convergence ratio
        csv_file = f"../log/{self.share_file_name}.csv"
        write_header = not os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["iteration", "conv", "remaining", "total"])
            writer.writerow([self.iter, converged, not_converged, total])

    @staticmethod
    def _average_and_normalize(original_emb, positive_texts, emb_dict):
        """Average original embedding with positives from emb_dict and L2 normalize."""
        emb = original_emb.copy()
        for text in positive_texts:
            emb += emb_dict[text]
        count = 1 + len(positive_texts)  # original embedding + positives
        emb /= count
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def _extend_dimension(self):
        """
        The goal is to increase dimension of the sparse vector if text is not similar to representative text recognized by LLM
        """
        # Find texts with no embedding yet
        none_texts = [text for text in self.test_examples_txt if self.text_to_sparse_vec.get(text) is None]
        print(f'There are {len(none_texts)} none-categorized')
        if not none_texts:
            return 

        
        # Texts with existing embeddings
        existing_texts = [text for text in self.test_examples_txt if self.text_to_sparse_vec.get(text) is not None]
        existing_embeddings = np.array([self.text_to_sparse_vec[text] for text in existing_texts])
        one_hot_none = np.eye(len(none_texts))
        
        # Pad existing embeddings with zeros for new dimensions
        existing_padded = np.hstack([
                                    existing_embeddings,
                                    np.zeros((existing_embeddings.shape[0], len(none_texts)))
                                    ])
        none_padded = np.hstack([
            np.zeros((len(none_texts), existing_embeddings.shape[1])),
            one_hot_none
        ])

        final_embeddings = np.vstack([existing_padded, none_padded])

        
        # Update self.text_to_sparse_vec with new extended embeddings
        for i, text in enumerate(existing_texts):
            self.text_to_sparse_vec[text] = final_embeddings[i]

        for i, text in enumerate(none_texts):
            self.text_to_sparse_vec[text] = final_embeddings[len(existing_texts) + i]




        

    def _save_text_embeddings(self,filepath):
        # Extract texts and embeddings in the same order
        texts = list(self.text_to_sparse_vec.keys())
        gd_labels = [self.text_to_gd_label[text] for text in texts]
        dense_embeddings = np.array([self.text_to_dense_vec[t] for t in texts])
        sparse_embeddings = np.array([self.text_to_sparse_vec[t] for t in texts])

        # Save embeddings to npz (compressed)
        np.savez_compressed(filepath + 'D.npz', embeddings=dense_embeddings)
        np.savez_compressed(filepath + 'S.npz', embeddings=sparse_embeddings)

        # Save texts to json
        formatted_data = [
            {
                'text': text,
                'Ground Truth': label
            }
            for text, label in zip(texts, gd_labels)
        ]

        with open(filepath + '.json', 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    def initial_grouping(self):
                
        if self.initial_dimension == 'full':
            cluster_num = len(self.test_examples_txt)
            cluster_labels = [i for i in range(cluster_num)]

        else:
            cluster_num = min(int(self.initial_dimension),len(self.test_examples_txt))
            
            if self.initial_cluster == 'agg':
                agg = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward') # AgglomerativeClustering
                cluster_labels = agg.fit_predict(self.embeddings)

            elif self.initial_cluster == 'kmeans':
                kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)  # n_init=10 is default
                cluster_labels = kmeans.fit_predict(self.embeddings)

        cluster_to_indices = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            cluster_to_indices[label].append(idx)

        # Find medoid and collect texts and embeddings
        cluster_to_representative_text = {} # map cluster to represent text
        cluster_to_embeddings = {} # map cluster to represent embd
        sampled_gd_labels = [] # map cluster to gd label of the represent text

        for cluster_id, indices in cluster_to_indices.items():
            cluster_embeddings = self.embeddings[indices]  # Subset of embeddings in this cluster
            
            dists = pairwise_distances(cluster_embeddings, metric='cosine')
            total_dists = dists.sum(axis=1)
            medoid_idx = indices[np.argmin(total_dists)]
            cluster_to_representative_text[cluster_id] = self.test_examples_txt[medoid_idx]
            sampled_gd_labels.append(self.test_examples_label[medoid_idx])
            cluster_to_embeddings[cluster_id] = self.embeddings[medoid_idx]

        # derive labeled text
        sorted_cluster_ids = sorted(cluster_to_representative_text.keys())
        self.representative_texts = [cluster_to_representative_text[k] for k in sorted_cluster_ids]
        self.representative_embeddings = np.stack([cluster_to_embeddings[k] for k in sorted_cluster_ids])

         

        # Build a sparse vector


        for i, text in enumerate(self.representative_texts):
            vec = np.zeros(len(self.representative_texts))
            vec[i] = 1  
            self.text_to_sparse_vec[text] = vec
        print(f'Initial cluster number: {len(self.representative_texts)}; Real clusters involved {len(set(sampled_gd_labels))}')





parser = argparse.ArgumentParser(description='LLM retrevial')
parser.add_argument('--seed_value', type=int,default = 1)
parser.add_argument('--dataset_name', type=str, default = 'banking')



parser.add_argument('--retriever_name', type=str,default = 'sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--llm_name',default = "google/gemma-2-9b-it")
parser.add_argument('--budget_num', type=int,default = 10)
parser.add_argument('--partition_num', type=int,default = 3)
parser.add_argument('--initial_cluster', type=str, default = 'agg')
parser.add_argument('--initial_dimension', type=str, default='2048')




args = parser.parse_args()
seed_value = args.seed_value
retriever_name = args.retriever_name
llm_name = args.llm_name
budget_num = args.budget_num
partition_num = args.partition_num
initial_cluster = args.initial_cluster
dataset_name = args.dataset_name
initial_dimension = args.initial_dimension

# Reproducible
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  
np.random.seed(seed_value)
random.seed(seed_value) 

# import data             




# Inference

start_time = time.time()
assign_cluster = InContextLLM(retriever_name = retriever_name,
                              llm_name = llm_name,
                              dataset_name = dataset_name,
                              budget_num = budget_num,
                              partition_num = partition_num,
                              initial_cluster = initial_cluster,
                              initial_dimension = initial_dimension,
                              seed_value = seed_value)

assign_cluster.get_embeddings()
assign_cluster.initial_grouping()

if initial_dimension != 'full':
    assign_cluster.group_with_llm(mode = 'initial')
    
for i in range(10):
    assign_cluster.group_with_llm(mode = 'iterative')



end_time = time.time()
print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")
