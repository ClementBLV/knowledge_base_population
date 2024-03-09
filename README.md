# Textual Entailment for Link Prediction in WordNet


Official code repository for ESWC 2024 paper 
"Textual Entailment for Link Prediction in WordNet". 

The paper is available at [https:/lien .pdf](https://.pdf).

Clément BELIVEAU - [IMT Atlantique](https://www.imt-atlantique.fr/en) 

[Guillermo Echegoyen Blanco]() and
[José Manuel Gómez-Pérez](https://scholar.google.com/citations?user=P3B2MmwAAAAJ&hl=fr&oi=ao) - [Exper.ai](https://www.expert.ai)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)


<html> 
<body> 
	<div class="image-container"> 
		<img src="https://www.pole-emc2.fr/app/uploads/logos_adherents/91fff3f6-c993-67c6-68ae-53957c2f623d-768x522.png" alt="Image 1" height="200">
		<img src="https://www.expert.ai/wp-content/uploads/2020/09/logo-new.png" alt="Image 2" height="100"> 
	</div> 
</body>
</html>

Presentation of the paper 

## Installation 🛠 

### Conda venv

```[bash]
   git clone https://github.com/expertailab/.git
   cd 
   python3 -m venv ~//venvFicus
   source ~/venvFicus/bin/activate
   pip install -r requirement.txt
```
### Conda env 

```[bash]
   git clone 
   cd 
   conda create -n KBentail python=3.9
   conda activate KBentail
   pip install -r requirement.txt
```
### Pipeline 

import pipeline image 

## Get started 🚀

### Dataset 

For all our experiments we have used three datasets: WR18RR dataset, available in the `data` file. The code to preprocess the data was heavily inspired by the code of the official repo of [SimCKG](https://github.com/intfloat/SimKGC) paper. 
### Models 

We have used `deberta-dase-v3` as our model, all models used are available on hugging face. We have two pre-trained models, one with prior training on entailment task name `"MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"` ([here]()) and another one without `"microsoft/deberta-v3-base"` ([here]())

### Run scipts 
- *Run training*

All the training was done on GTXForce 1060, the hyperparameters are available in the section hyperparameters of the appendix ([here]()). To train our model we have adapted the script `run_glue`. given by hugging face. Before running the scripts, you can either activate your conda environment with the command: `conda activate KBentail`, or you can directly precise the path of your venv in the second line of the scripts `source //PATH_TO_YOUR_VENV/YOUR_VENV/bin/activate`

```[bash]
	source script_train.sh 
```

- *Run inference*

```[bash]
	source script_eval.sh 
```

- *Run evaluations*

```[bash]

```

### Expected results : 

give the results from the appendix 
give the appendix en pdf sur le github


## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
bibtex
```
