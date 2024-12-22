# Textual Entailment for Link Prediction in a knowledge base

![Conference](https://img.shields.io/badge/Conference-EKAW%202024-red)
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to the official code repository for the ESWC 2024 paper **"Textual Entailment for Link Prediction in WordNet"**.

- **Paper:** [Available here](https://.pdf)
- **Authors:** Clément BELIVEAU ([IMT Atlantique](https://www.imt-atlantique.fr/en)), Guillermo Echegoyen Blanco, and José Manuel Gómez-Pérez ([Expert.ai](https://www.expert.ai))
<html>
<body>
    <div class="center">
        <img src="https://github.com/ClementBLV/knowledge_base_population/blob/main/doc/logos.png" alt="Logos">
    </div>
</body>
</html>

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Conda Virtual Environment](#conda-virtual-environment)
- [Usage](#usage)
  - [Datasets](#datasets)
  - [Models](#models)
- [Run Training](#run-training)
  - [Bash](#bash)
  - [Docker](#docker)
  - [Colab](#colab)
- [Results](#results)
- [Citation](#citation)

## Introduction
This repository contains the code for the paper **"Textual Entailment for Link Prediction in WordNet"**. Our approach leverages the abstraction capabilities of Large Language Models (LLMs) to tackle link prediction within a knowledge base, specifically WN18RR. Our method demonstrates state-of-the-art performance using a base-size model.

## Installation 🛠

### Prerequisites
- Python installed
- SSH key of your GitHub account configured

### Conda Virtual Environment

#### Using `venv`
```bash
git clone git@github.com:ClementBLV/knowledge_base_population.git
cd knowledge_base_population
python3 -m venv ~/EntKBC
source ~/EntKBC/bin/activate
pip install -r requirements.txt
```

#### Using `conda`
```bash
git clone git@github.com:ClementBLV/knowledge_base_population.git
cd knowledge_base_population
conda create -n EntKBC python=3.9
conda activate EntKBC
pip install -r requirements.txt
```

In case you encounter issues during the execution you can run : 
```bash 
bash executable.sh
```

## Usage 🚀

### Datasets
For all our experiments, we use the WN18RR dataset, available in the `data` folder. The data preprocessing code is adapted from the [SimCKG](https://github.com/intfloat/SimKGC) repository.

### Models
We use `deberta-base-v3` as our model. The models are available on Hugging Face:
- Pre-trained on entailment tasks: [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
- Without prior training: [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base)

### Pipeline 

<CENTER>
<img
src="https://github.com/ClementBLV/knowledge_base_population/blob/main/doc/sch%C3%A9ma_entailement.jpg"
WIDTH=100%>
</CENTER>

## Run Training 🏃

---
<CENTER>
<u><b>BASH</b></u>
</CENTER>


---

### Base models : 

This script will help you to train an entailment model on a relation classification task. Step by step command are detailed to understand the different parts of the script

```bash
source script_train_expert.sh \
  --split int \ #percentage of the data which will be used 
  --both boolean \
  --bias boolean \
  --processed_data_directory "/your/directory/" #directory of the data to process usually knowledge_base_population/data/WN18RR/ but data folder can be moved in another location without an issue
  --model "hugging face id" \ #name of the model wich will be used for training
  --output_dir "/your/directory" \ #where the training output will be stored
  --task "" \ # either wn/wordnet/wn18rr or fb/freebase/fb15k237 
  --no_training boolean \ # it the training is launched or not
  --hf_cache_dir "//users/local/c20beliv/" #hugging face cache dir 
  --config_file config.json #config file for your training parameters located in configs folder
```
- **Step 1 :** Firstly lets do a dummy test without any training to check that the data are correctly processed and save in the folders. After starting the venv, copy and paste :  
    ```bash
    source script_train_expert.sh --split 1   --both false   --bias true   --processed_data_directory ~/knowledge_base_population/data/WN18RR/  --model "microsoft/deberta-v3-small"   --output_dir ~/knowledge_base_population/data/WN18RR/weights   --task "wn"   --no_training true   --hf_cache_dir //users/local/c20beliv/  --config_file "config.json" --wandb_key ""
    ```

- **Step 2:** Now try it with Freebase dataset
    ```bash
    source script_train_expert.sh --split 1 --both false   --bias true   --processed_data_directory ~/knowledge_base_population/data/FB15k237/ --model "microsoft/deberta-v3-small"   --output_dir ~/knowledge_base_population/data/FB15k237/weights   --task "fb"  --no_training true   --hf_cache_dir //users/local/c20beliv/  --config_file "config.json" --wandb_key ""
    ```

- **Step 3:** Try to train the model with hugging face
    ```bash
    source script_train_expert.sh   --split 1 --both false   --bias true   --processed_data_directory ~/knowledge_base_population/data/FB15k237/ --model "microsoft/deberta-v3-small"   --output_dir ~/knowledge_base_population/data/FB15k237/weights --task "fb" --no_training false   --hf_cache_dir //users/local/c20beliv/ --config_file "config.json" --wandb_key ""
    ```

- **Step 4:** Evaluation 
    ```bash
    source script_eval_expert.sh --task "wn" --weights_path "//users/local/c20beliv/MNLI_wn/naive_derberta_small_2w_biased_split1_v-20241120" --saving_name "eval1" --parallel "true" --batch_size 3 --hf_cache_dir "//users/local/c20beliv/" --model "microsoft/deberta-v3-small"
    ```

---- 

### Meta model: 

1. **Binairy Prediction**: 

  The input for binary prediction is a matrix where each column corresponds to a model's prediction probabilities. Each column contains two probabilities: one for entailment (`p_{entail}`) and one for contradiction (`p_{contra}`), as described below:

  $$
  \mathbf{X (tail, relation, head)} = \mathbf{X (hypostesis, premise)} = 
  \begin{bmatrix}
  p_{1,\text{entail}}  & p_{2,\text{entail}} & p_{3,\text{entail}} & p_{4,\text{entail}}
  \end{bmatrix}
  $$

  $$
  \mathbf{Y} = real \space label  = {0=entailement , 1= contradiction}
  $$

  UPDATE --> only keep the p_entail hense 4 probabilites
  for data in data (batch - 1000): 
    compute the 4 prob 
    train meta model 
    if loss decrees 
      continue 
    else: 
      stop

first performace = 0 
for i in step : 
  start with (step * 10%) : 
  generate the data to train (4 probabilities)
  train the model 
  evaluate 
  if new performance > first_perf : 
    continue 
  else : 
    stop 

----> heuristic, improvement of epsilon choosen ( eg improvement below 10-3)



  Here:
  - $( p_{i,\text{entail}} )$ is the probability of entailment from the $(i)$-th model.
  - $( p_{i,\text{contra}} )$ is the probability of contradiction from the $(i)$-th model.
  - The matrix has dimensions $(1 \times 4 \times 2)$, where the row represent the probabilities for the entailment and contradiction classes for a given couple of hypothesis an premise for a given relation, and the columns represent the four models.

  Then given $X$ the model is trained to predict the probability of entailment or contradiction of the pair of hypothesis and premise

  $$
  \text{BinairyModel}(X) = \{0, 1\}
  $$
  ---
  Here:
  - $0$ mean an entailment hense the head and tail are linked by the relation.
  - $1$ means a contradiction  hense the head and tail are not linked by the relation.

  Code : 

  ```bash 
  source script_train_meta_binary_expert.sh --task "wn" --processed_data_directory ~/knowledge_base_population/data/WN18RR/ --input_file "train.json" --training_file_name "train_meta.mnli.json" --weight_dir "//users/local/c20beliv/" --saving_name "meta_wn_20241126.json" --no_training "true" --parallel "false"  --config_file "config_meta.json"
  ```
  For binairy classification `script_train_meta_binary_expert.sh`, starts with `data2mnli.py` on the data of your choise (train , test, valid) precised as the `--input_file` located in the processed data directory. Then, ones converted into the right format of MNLI data we need for the meta model it will call `data2meta.py` to compute the probability using each model. If you ask for training a meta model will be trained on those data. Ones the preprocessing has been done, you can skip this part using the argument `--do_preprocess` on false. 

  ```bash 
  source script_train_meta_expert.sh --task "wn" --processed_data_directory ~/knowledge_base_population/data/WN18RR/ --input_file "train.json" --training_file_name "train_meta.mnli.json" --weight_dir "//users/local/c20beliv/" --saving_name "meta_wn_20241126.json" --no_training false --parallel "false"  --config_file "config_meta.json" --do_preprocess false
  ```

2. **Vector Prediction**: 

  With this method we are doing a more refined prediction where the model is given all the probability  one for entailment (`p_{entail}`) and one for contradiction (`p_{contra}`) obtained with the three models, for each relation at ones. Then it must output a vector of size [1, Number of relation], for instance the output is [0, 0 , 1, 0, 0] if the relation 5 is the good relation. 

  $$
  \mathbf{X ( head ,relation_{i \in \{0, 3\}}, tail )} = 
  \begin{bmatrix}
  p_{1,\text{entail}}^{r1} & p_{2,\text{entail}}^{r1} & p_{3,\text{entail}}^{r1} & p_{4,\text{entail}}^{r1} \\
  \\
  p_{1,\text{entail}}^{r2} & p_{2,\text{entail}}^{r2} & p_{3,\text{entail}}^{r2} & p_{4,\text{entail}}^{r2} \\
  \\
  p_{1,\text{entail}}^{r3} & p_{2,\text{entail}}^{r3} & p_{3,\text{entail}}^{r3} & p_{4,\text{entail}}^{r3} 
  \end{bmatrix}
  $$

  Here:
  - $( p_{i,\text{entail}}^{ri} )$ is the probability of entailment from the $(i)^{th}$ model for the $(i)^{th}$ relation.
  - relation_{i \in \{0, 3\}} all the relation in the dataset, each row of the input matrix correspond to one of them
  - The matrix has dimensions $(n \times 4 \times 2)$, where the row represent the probabilities for the entailment and contradiction classes for a given couple of hypothesis an premise for all relation, and the columns represent the four models.

  Then given $X$ the model is trained to predict the probability of entailment or contradiction of the pair of hypothesis and premise for all the relations

  $$
  \text{VectorModel}(X) = 
  \begin{bmatrix}
  0 & 0 & 1
  \end{bmatrix}
  $$

  Here:
  - The matrix has dimensions $(1 \times n)$

  Code: 
  

  ```bash 
  source script_train_meta_expert.sh --task "wn" --processed_data_directory ~/knowledge_base_population/data/WN18RR/ --input_file "train.json" --training_file_name "train_meta.mnli.json" --weight_dir "//users/local/c20beliv/" --saving_name "meta_wn_20241126.json" --no_training "true" --parallel "false"  --config_file "config_meta.json"
  ```

---
<CENTER>
<u><b>DOCKER</b></u>
</CENTER>

---

To build and run the Docker containers:
1. Build Docker containers:
   ```bash
   sh build_docker.sh
   ```
    This command will build six docker container each one corresponding to a different split (1; 5; 7; 10; 20 and 100 %). If you only want one container with a given split value you should consider to modify the for loop in this file. 
2. Run Docker containers:
   ```bash
   docker run -v `pwd`/volume split_1 [--no_training]
   ```
   As in the section above the argument `--no_training` means that no training will be done, it is usefull to check the installation and the build was done correctly. To run the training you should remove it.  

---
<CENTER>
<u><b>Colab</b></u>
</CENTER>

---
You can also run the training in Google Colab. Click the icon below to open the notebook:

[![Colab](https://img.shields.io/static/v1?label=Google&message=Open+with+Colab&color=blue&style=plastic&logo=google-colab)](https://colab.research.google.com/drive/16FhTJkbedGBKlUVF52hH3enEybqaWkhN#scrollTo=maNLM25S2i8q)

## Results ✌️
The expected results and detailed evaluations are provided in the appendix, available [here](https://github.com/ClementBLV/knowledge_base_population/doc/blob/main/Entailement_Paper___appendix.pdf).

## Citation
If you find our paper or code repository helpful, please consider citing as follows:

```bibtex
@article{beliveau2024textual,
  title={Textual Entailment for Link Prediction in WordNet},
  author={Beliveau, Clément and Blanco, Guillermo Echegoyen and Gómez-Pérez, José Manuel},
  journal={EKAW 2024},
  year={2024}
}
```
