
from transformers import AutoTokenizer, DebertaForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

path = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(path)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
model.to(device)

premise="Portsmouth means a port city in southeastern Virginia on the Elizabeth River opposite Norfolk; naval base; shipyards. city or metropolis means a large and densely populated urban area; may include several independent administrative districts"
hypothesis_true= "Portsmouth is a city or metropolis."

input = tokenizer(premise,hypothesis_true, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
entailment = [prediction[0]]  # the True element is at index 0
print(entailment)

path = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(path)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
model.to(device)

input = tokenizer(premise,hypothesis_true, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
entailment = [prediction[0]]  # the True element is at index 0
print(entailment)

path = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(path)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
model.to(device)
input = tokenizer(premise,hypothesis_true, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
entailment = [prediction[0]]  # the True element is at index 0
print(entailment)
input = tokenizer(premise,hypothesis_true, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
entailment = [prediction[0]]  # the True element is at index 0
print(entailment)