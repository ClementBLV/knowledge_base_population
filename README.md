# Textual Entailment for Link Prediction in a knowledge base

![Conference](https://img.shields.io/badge/Conference-EKAW%202024-red)
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to the official code repository for the ESWC 2024 paper **"Textual Entailment for Link Prediction in WordNet"**.

- **Paper:** [Available here](https://.pdf)
- **Authors:** Clément BELIVEAU ([IMT Atlantique](https://www.imt-atlantique.fr/en)), Guillermo Echegoyen Blanco, and José Manuel Gómez-Pérez ([Expert.ai](https://www.expert.ai))


<html> 
<body> 
	<div class="image-container" style="display: flex; align-items: center;"> 
		<img src="https://www.pole-emc2.fr/app/uploads/logos_adherents/91fff3f6-c993-67c6-68ae-53957c2f623d-768x522.png" alt="Image 1" height="100" style="margin-right: 20px;">
		<img src="https://www.expert.ai/wp-content/uploads/2020/09/logo-new.png" alt="Image 2" height="50" style="margin-top: -35px;"> 
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
python3 -m venv ~/venvFicus
source ~/venvFicus/bin/activate
pip install -r requirements.txt
```

#### Using `conda`
```bash
git clone git@github.com:ClementBLV/knowledge_base_population.git
cd knowledge_base_population
conda create -n KBentail python=3.9
conda activate KBentail
pip install -r requirements.txt
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

### Bash
```bash
source script_train_expert.sh \
  --split 1 \
  --both false \
  --bias true \
  --processed_data_directory ~/knowledge_base_population/data/FB15k237/ \
  --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" \
  --output_dir ~/knowledge_base_population/data/FB15k237/weights \
  --task "fb" \
  --no_training false #(to check the installation) / true #(to launch the training)
```

### Docker
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

### Colab
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