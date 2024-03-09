# Textual Entailment for Link Prediction in WordNet


Official code repository for ESWC 2024 paper 
"Textual Entailment for Link Prediction in WordNet". 

The paper is available at [https:/lien .pdf](https://.pdf).

[IMT Atlantique](https://www.imt-atlantique.fr/en) 

Clément BELIVEAU


[Exper.ai](https://www.expert.ai)

[Guillermo Echegoyen Blanco]() and
[José Manuel Gómez-Pérez](https://scholar.google.com/citations?user=P3B2MmwAAAAJ&hl=fr&oi=ao)

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
   git clone https://github.com/expertailab/.git
   cd 
   conda create -n Ficus python=3.9
   conda activate Ficus
   pip install -r requirement.txt
```
### Pipeline 

import pipeline image 

## Get started 🚀

### Dataset 

For all our experiments we have used three datasets  : WR18RR dataset, available in the `data` file. The code to preprocess of the data was heavily inspired from the code of the official repo of [SimCKG](https://github.com/intfloat/SimKGC) paper. 
### Models 

We have used `deberta-dase-v3` as our model. 

### Train the model 


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

### Results : 

give the results from the appendix 
give the appendix en pdf sur le github


## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
bibtex
```
