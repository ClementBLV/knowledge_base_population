# Textual Entailment for Link Prediction in WordNet


Official code repository for ESWC 2024 paper 
"Textual Entailment for Link Prediction in WordNet". 

The paper is available at [https:/lien .pdf](https://.pdf).

[IMT Atlantique](https://www.imt-atlantique.fr/en) 

Clément BELIVEAU


[Exper.ai](https://www.expert.ai)

Guillermo Echegoyen Blanco and
José Manuel Gómez-Pérez

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)


<html> 
	<head> 
		<style> .image-container { 
			display: flex;
			 justify-content: center; 
			 } 
			 .image-container img { 
			 margin: 0 10px; /* Adjust the margin as needed */ height: 200px; /* Set the height, width will scale proportionally */ }
         </style> 
	</head>
<body> 
	<div class="image-container"> 
		<img src="https://www.pole-emc2.fr/app/uploads/logos_adherents/91fff3f6-c993-67c6-68ae-53957c2f623d-768x522.png" alt="Image 1" height="200">
		<img src="https://www.expert.ai/wp-content/uploads/2020/09/logo-new.png" alt="Image 2" height="100"> 
	</div> 
</body>
</html>


This repository contains the code for out of the box ready to use few-shot classifier for ambiguous images. In this paper we have shown that removing the ambiguity from the the query during few shot classification improves performances. To do so we use a combination of foundation models and spectral methods. 
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
   git clone https://github.com/expertailab/.git
   cd 
   conda create -n Ficus python=3.9
   conda activate Ficus
   pip install -r requirement.txt
```
## Pipeline 



## Get started 🚀

### Dataset 

For all our experiments we have used three datasets  : WR18RR dataset, available in the `dataset` file
### Models 

We have used `deberta-dase-v3` as our model. 

### Run inference

- To run the evaluations  
```[bash]
sh run.sh -dataset "cub"
```
- To run deep spectral method on un image
```[bash]

```



- To run prompted sam on an image
```[bash]

```


## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
bibtex
```
