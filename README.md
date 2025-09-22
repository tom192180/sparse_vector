

# Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs

## Introduction

### What is our work?

We propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder.

### What is good about it?
Results are comparable with other SOTA approaches. However, our method requires only low resources, makes fewer assumptions, is easier to incorporate with other approaches, and is scalable.

## Follow the following steps to reproducible the results: 

### Step 1: Download related datasets

**Emerging scenario:**

* **Mtop**, **Massive**, **GoEmo**: We use the same dataset settings as one of baselines we compare, ClusterLLM: https://github.com/zhang-yu-wei/ClusterLLM

* **clinc150**: https://github.com/clinc/oos-eval (This can be downladed from ClusterLLM as well, but we oringally downloaded from here)

* **Bank77**: Huggingface (This can be downladed from ClusterLLM as well, but we oringally downloaded from here)

* **Update the dataset filepaths in** `cluster_dataset.py`: Transform all datasets from .json into CSV format with 'text,label' per row and no header. The only exception is Bank77 from HuggingFace, which can be used as is. 

**GCD scenario:**

* **clinc150**, **Bank77**, **Stackoverflow**: data can be downloaded from here, https://github.com/liangjinggui/ALUP/tree/main/data

* **Update the dataset filepaths in** `cluster_dataset.py`: This files are tsv files,  `cluster_dataset.py` processed them automatically. Only to update these tsv filepaths in the script.

### Step 2: Install the required package from the `requirements.txt`
```
  pip install -r requirements.txt
```

### Step 3: Update the huggingface token, and filepaths in `run.py`: 
* Some models, such as LLaMA, require your hg token. Update it on lines 83 and 86. Also, update the file paths on lines 252 and 428 to save the convergence ratio CSV and the updated sparse vector. 

* (Optional): Uupdate line 97 if you use instructor-large as the backbone embedder. There is a version incompatibility between sentence-transformers and the HuggingFace libraries we use, but it can be fixed by first downloading the instructor model locally.

### Step 4: Update the filepath in `eval.py`: 
Update it on line 202 to save the evaluation results.



### Step 5: Get the sparse vectors.

Nice! Once you finish the first four steps, you can start reproducing our work.
Run the following to get the sparse vectors.

```python
  python run.py\
    --seed_value 1 \
    --dataset_name banking \
    --partition_num 3 \
    --retriever_name sentence-transformers/all-MiniLM-L6-v2\
    --llm_name google/gemma-2-9b-it \
    --initial_dimension 2048 \
```
* Other possible dataset_name: clinc, massive, mtop, emo, clincKIR, bankingKIR, stackoverflowKIR.  Datasets ending with KIR (Known intent/category ratio) are for the **GCD scenario**. In our work, we assume none of the intents/categories are known; the reamainings are for **Emerging data scenario**.
* Other possible retriever_name: hkunlp/instructor-large, intfloat/e5-large, google-bert/bert-base-uncased
* Other possible llm_name: Qwen/Qwen3-8B, meta-llama/Meta-Llama-3.1-8B-Instruct


### Step 6: Evaluate the sparse vector.

```python
  python /home/ilin/p4_intentCr_known/script/eval_hybrid.py --file_path "$file_path" --method H
```
* The file_path is the file end with .json.  .npz will be load directly by the script based on the json file you give.
* Other possible method: H means HDBSCAN, K means Kmeans.

