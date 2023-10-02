import torch

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer


import pandas as pd

## code inspired from https://towardsdatascience.com/fine-tuning-pre-trained-transformer-models-for-sentence-entailment-d87caf9ec9db


class Model_loader:
    def __init__(
        self , hypo_path , hyper_path
    ):
        if hypo_path is not None:
            self.model_hypo = self.load_hypo(hypo_path)
        if hyper_path is not None:
            self.model_hyper = self.load_hyper(hyper_path)

    def predict(self, tokenized_sentense, model_type):
        assert (
            type(tokenized_sentense) != str
        ), "The sentence for the prediction must be tokenized"
        assert model_type in [
            "hypo",
            "hyper",
        ], "The model type should be an hyponym or an hypenym"
        if model_type == "hypo":
            model = self.model_hypo
        if model_type == "hyper":
            model = self.model_hyper
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_size, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(
                tokenized_sentense
            ):
                # optimizer.zero_grad()
                pair_token_ids = pair_token_ids
                seg_ids = seg_ids

                prediction = model(
                    pair_token_ids,
                    token_type_ids=seg_ids,
                ).values()
                predictions.append(list(prediction)[0])
        return [int(torch.argmax(t, dim=1)) for t in predictions]

    def load_hypo(self, output_model_hypo: str):
        # load
        checkpoint_hypo = torch.load(output_model_hypo, map_location="cpu")
        model_hypo = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model_hypo.load_state_dict(checkpoint_hypo["model_state_dict"], strict=False) 
        # strict add to get rid of the pb with dict key due to the download from drive 
        param_optimizer = list(model_hypo.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        optimizer_hypo = AdamW(
            optimizer_grouped_parameters, lr=2e-5, correct_bias=False
        )

        optimizer_hypo.load_state_dict(checkpoint_hypo["optimizer_state_dict"])
        return model_hypo

    def load_hyper(self, output_model_hyper: str):
        # load
        checkpoint_hyper = torch.load(output_model_hyper, map_location="cpu")
        model_hyper = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model_hyper.load_state_dict(checkpoint_hyper["model_state_dict"],strict=False) 


        param_optimizer = list(model_hyper.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        optimizer_hyper = AdamW(
            optimizer_grouped_parameters, lr=2e-5, correct_bias=False
        )
        optimizer_hyper.load_state_dict(checkpoint_hyper["optimizer_state_dict"])

        return model_hyper


class Tokenizer:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )  # Using a pre-trained BERT tokenizer to encode sentences

    def tokenize(self, hypothesis: str, premise: str):
        """Tokenize all the element to put them into the corect form for the model
        predict the entailement

        Args:
            hypothesis (list(str)): hypothesis should follow the model
                            target + candidate (not the other order)
            premise (list(str)): sentence to which we are going to check if the
                            hypothesis holds
        """
        assert len(hypothesis) == len(
            premise
        ), "The two list must contain the same number of items"
        dataframe1 = pd.DataFrame(columns=["sentence1", "sentence2", "gold_label"])
        dataframe1["sentence1"] = [""]
        dataframe1["sentence2"] = [""]
        dataframe1["gold_label"] = ["entailment"]
        dataframe1

        dataframe2 = pd.DataFrame(columns=["sentence1", "sentence2", "gold_label"])
        dataframe2["sentence1"] = hypothesis
        dataframe2["sentence2"] = premise
        dataframe2["gold_label"] = ["entailment" for i in hypothesis]
        dataframe2

        mnli_dataset = MNLIDataBert(dataframe1, dataframe2, self.tokenizer)
        _, val_loader = mnli_dataset.get_data_loaders(batch_size=1, shuffle=False)
        return val_loader


class MNLIDataBert(Dataset):
    def __init__(self, train_df, val_df, tokenizer):
        self.label_dict = {"entailment": 0, "contradiction": 1}  # , 'neutral': 2}

        self.train_df = train_df
        self.val_df = val_df

        self.base_path = "/content/"
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)

    def load_data(self, df):
        MAX_LEN = 512
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premise_list = df["sentence1"].to_list()
        hypothesis_list = df["sentence2"].to_list()
        label_list = df["gold_label"].to_list()

        for premise, hypothesis, label in zip(
            premise_list, hypothesis_list, label_list
        ):
            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = (
                [self.tokenizer.cls_token_id]
                + premise_id
                + [self.tokenizer.sep_token_id]
                + hypothesis_id
                + [self.tokenizer.sep_token_id]
            )
            premise_len = len(premise_id)
            hypothesis_len = len(hypothesis_id)

            segment_ids = torch.tensor(
                [0] * (premise_len + 2) + [1] * (hypothesis_len + 1)
            )  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor(
                [1] * (premise_len + hypothesis_len + 3)
            )  # mask padded values

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            y.append(self.label_dict[label])

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        train_loader = DataLoader(
            self.train_data, shuffle=shuffle, batch_size=batch_size
        )

        val_loader = DataLoader(self.val_data, shuffle=shuffle, batch_size=batch_size)

        return train_loader, val_loader


if __name__ == "__main__":
    """model = Model_loader(hypo_path="model_hyponyms.pth")
    tokenizer = Tokenizer()

    amphibian = "amphibian a cold-blooded vertebrate animal of a class that comprises the frogs, toads, newts, salamanders, and caecilians. "
    frog = "frog a tailless amphibian with a short squat body, moist smooth skin, and very long hind legs for leaping."
    ranid = "ranid is any of a large family (Ranidae) of long-legged frogs distinguished by extensively webbed hind feet, horizontal pupils, and a bony sternum. "
    wood_frog = "wood-frog  a common North American frog that is found mostly in moist woodlands and is dark brown, yellowish brown, or pink with a black stripe on each side of the head."

    hypothesis = ranid + frog
    premises = "ranid specifies frog"

    hypothesis2 =   frog + amphibian
    premises2  = "amphibian specifies frog"

    input = tokenizer.tokenize(hypothesis=[hypothesis, hypothesis2], premise=[premises, premises2])
    print(model.predict(input, model_type="hypo"), file=sys.stderr)"""
