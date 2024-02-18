# Imports
import uuid, torch
import numpy as np
import pandas as pd
import pprint as pp
from ray import tune
from ray.air import session
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (accuracy_score, 
                            precision_score, 
                            recall_score, 
                            f1_score,
                            roc_auc_score)

class LoraModel():

    # Initialize the LoraModel with a DataDict formatted dataset and 
    #   keyword arguments that align with those found in "get_parameters()"
    def __init__(self, dataset, **kwargs):
        self.results = None
        self.params = self.get_parameters(kwargs)
        self.dataset = self.preprocess_dataset(dataset)
        if self.params["load_model"] is not None:
            self.load_model(self.params["load_model"])
        else: self.model = self.build_model()

    # Dataset preprocessing and tokenization:
    #   Dataset is assumed to be huggingface-formatted and 
    #   of form {"train": {"X":[], "Y":[]}, "test": {"X":[], "Y":[]}}
    def preprocess_dataset(self, dataset):
        self.texts = dataset["test"]["X"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model_type"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized_dataset = dataset.map(
            lambda example: self.tokenizer(example["X"], max_length=self.params["maxtokens"], padding='max_length', truncation=True),
            batched=True
        )
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Y'])
        tokenized_dataset = tokenized_dataset.remove_columns(["X"]).rename_column("Y", "labels")
        return tokenized_dataset

    # Setup a new model from HuggingFace, PEFT, Lora if none is loaded
    def build_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.params["model_type"], num_labels=2)
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, 
                                    r=self.params["lora_r"], 
                                    lora_alpha=self.params["lora_alpha"], 
                                    lora_dropout=self.params["lora_dropout"])
        model = get_peft_model(model, peft_config)
        model = model.to("cuda")
        return model

    # Training function - Always uses AdamW
    def train(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.params["lr"])
        train_dataloader = DataLoader(self.dataset["train"], shuffle=True, batch_size=self.params["batch_size"])
        for epoch in np.arange(self.params["epochs"]):
            for batch in train_dataloader:
                for key in batch:
                    batch[key] = batch[key].to("cuda")
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{self.params['epochs']}: Loss = {loss}")

    # Evaluation function - Calculate statistics on the test set
    def evaluate(self):
        actuals = self.dataset["test"]["labels"].tolist()
        test_dataloader = DataLoader(self.dataset["test"], shuffle=False, batch_size=len(actuals))
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].to("cuda")
            with torch.no_grad():
                testout = self.model(**batch)
        probs = testout.logits.softmax(dim=-1).detach().cpu().numpy().tolist()
        preds = [np.argmax(p) for p in probs]
        probs = [p[1] for p in probs]
        self.calculate_metrics(actuals, preds, probs)

    # Predict from a new set of texts
    def predict(self, texts):
        tokenized_texts = self.tokenizer(
            texts, max_length=self.params["maxtokens"], padding='max_length', truncation=True
        )
        for key in tokenized_texts: 
            tokenized_texts[key] = torch.tensor(tokenized_texts[key]).to("cuda")
        testout = self.model(**tokenized_texts)
        probs = testout.logits.softmax(dim=-1).detach().cpu().numpy().tolist()
        preds = [np.argmax(p) for p in probs]
        probs = [p[1] for p in probs]
        return preds, probs
    
    # Function to store metrics
    def calculate_metrics(self, acts, preds, probs):
        self.results = pd.DataFrame({
            "Text": self.texts,
            "Actual": acts,
            "Predicted": preds,
            "Probability": probs
        })
        self.metrics = {
            "acc": accuracy_score(acts, preds),
            "prec": precision_score(acts, preds),
            "rec": recall_score(acts, preds),
            "f1": f1_score(acts, preds),
            "roc": roc_auc_score(acts, preds)
        }

    def get_parameters(self, params):
        params.setdefault('model_type', 'roberta-base')
        params.setdefault('lr', 4.12e-5)
        params.setdefault('lora_r', 16)
        params.setdefault('lora_alpha', 16)
        params.setdefault('lora_dropout', 0.1)
        params.setdefault('epochs', 20)
        params.setdefault('batch_size', 1)
        params.setdefault('maxtokens', 100)
        params.setdefault('load_model', None)
        return params

    def save_results(self, file="./results/results.csv"):
        self.results.to_csv(file)
    
    def save_model(self, name="mod"):
        self.model.save_pretrained(f"./model/{name}.hf")
    
    def load_model(self, name="mod"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"./model/{name}.hf", num_labels=2
        ).to("cuda")

    def __repr__(self):
        if self.results is None:
            return f"<LoraModel>: Unevaluated with parameters \n{pp.pformat(self.params)}"
        else:
            return (
                "\nEvaluated <LoraModel> with results"
                f"\nAccuracy: {self.metrics['acc']}"
                f"\nPrecision: {self.metrics['prec']}"
                f"\nRecall: {self.metrics['rec']}"
                f"\nF1 Score: {self.metrics['f1']}"
                f"\nROC-AUC Score: {self.metrics['roc']}"
                "\n\nModel Parameters:"
                f"\n{pp.pformat(self.params)}"
            )

# A "tunable" function to be passed to RayTune
def tune_func(config):
    dataset = load_from_disk("/mnt/data/dot.hf")
    mod = LoraModel(
        dataset = dataset,
        model_type = config['model_type'],
        lr = config['lr'],
        lora_r = config['lora_r'],
        lora_alpha = config['lora_alpha'],
        lora_dropout = config['lora_dropout'],
        epochs = config['epochs'],
        batch_size = config['batch_size']
    )
    mod.train()
    mod.evaluate()
    session.report({
        "accuracy": mod.metrics["acc"], 
        "precision": mod.metrics["prec"], 
        "recall": mod.metrics["rec"],
        "f1": mod.metrics["f1"],
        "auroc": mod.metrics["roc"],
        "score": 2*mod.metrics["rec"] + mod.metrics["acc"]
    })
 
# Define tuning hyperparameters and execute the script
# Optuna is used to attempt to find the optimal values
TUNE_PARAM = "score"
config = {
    "model_type": tune.choice([
        "bert-large-uncased", 
        "roberta-large",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "llmrails/ember-v1"
    ]),
    "lr": tune.loguniform(1e-6, 1e-4),
    "lora_r": tune.choice([16]),
    "lora_alpha": tune.choice([8, 16, 32, 64]),
    "lora_dropout": tune.uniform(0.1, 0.3),
    "epochs": tune.randint(10, 40),
    "batch_size": tune.choice([1])
}
opt = OptunaSearch(metric=TUNE_PARAM, mode="max")
tuner = tune.Tuner(
    tune.with_resources(tune_func, {"cpu": 0, "gpu": 1}),
    tune_config=tune.TuneConfig(
        search_alg=opt,
        num_samples=20
    ),
    param_space=config
)
result = tuner.fit()
result.get_dataframe().to_csv(f"./{str(uuid.uuid1())}.csv")