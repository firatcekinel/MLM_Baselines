from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import torch
import wandb
from tqdm import tqdm 
import argparse
import time
from Preprocessor import preprocess, batch_preprocess, batch_preprocess2
import ast
#model_id = "VRLLab/TurkishBERTweet"
#model_id = "dbmdz/bert-base-turkish-cased"
#model_id = "FacebookAI/xlm-roberta-large"

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='VRLLab/TurkishBERTweet', type=str, help='model id')
parser.add_argument('--task', default='ner', type=str, help='sentiment/offensive/nli/ner')
parser.add_argument('--dataset', default='ner_tweet_tr', type=str, help='tsad/offenseval2020_tr/xnli/ner/ner_tweet_tr')
parser.add_argument('--cache_dir', default='/mnt/proj1/dd-23-122/.cache', type=str, help='model cache dir')
parser.add_argument('--skip_train', default=False, type=bool, help='skip to inference')
parser.add_argument('--wandb_proj_name', default="bertweet", type=str, help='wandb project name')
args = parser.parse_args()

model_id = args.model_id
model_name = model_id.split("/")[-1]
cache_dir = args.cache_dir
skip_train = args.skip_train
output_dir = "outputs/" + args.task + "/" 
wandb_proj_name = args.wandb_proj_name
file_name = f"{args.task}_{model_name}_{args.dataset}"

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
metric = evaluate.load("seqeval")
epoch = 2

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, add_prefix_space=True)
if args.task == "ner":
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
else:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    if args.model_id == 'VRLLab/TurkishBERTweet':
        return tokenizer(preprocess(examples["text"]), truncation=True)
    
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if args.task == "ner":
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {"accuracy": all_metrics["overall_accuracy"], "f1-macro": all_metrics["overall_f1"]}

    else: 
        predictions = np.argmax(predictions, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc, "f1-macro": f1_macro}

def tokenize_and_align_labels(examples):
    if args.model_id == 'VRLLab/TurkishBERTweet':
        tokenized_inputs = tokenizer(batch_preprocess(examples["tokens"]), truncation=True, is_split_into_words=True)
    else:
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if args.task == "sentiment":
    all_labels = [0,1,2]
    label_dict = {"Notr": 0, "Positive": 1, "Negative": 2}
    id2label = {0:"Notr", 1:"Positive", 2:"Negative"}
    label2id = {"Notr": 0, "Positive": 1, "Negative": 2}

    train_df = pd.read_csv("datasets/tsad/train.csv", sep=",", encoding="utf-8")
    test_df = pd.read_csv("datasets/tsad/test.csv", sep=",", encoding="utf-8")
    
    train_df = train_df[["text", "label"]]
    train_df.dropna(inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    train_df['label'] = train_df['label'].replace(label_dict)
    
    test_df = test_df[["text", "label"]]
    test_df.dropna(inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df['label'] = test_df['label'].replace(label_dict)

    # create train and validation splits
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
        })
    
    if args.model_id == "VRLLab/TurkishBERTweet":
        tokenized_dataset = dataset.map(lambda x: tokenizer(batch_preprocess2(x['text']), truncation=True, padding=True, max_length=512), batched=True).remove_columns(["text"])
    else:
        tokenized_dataset = dataset.map(preprocess_function, batched=True).remove_columns(["text"])

elif args.task == "offensive":
    all_labels = [0,1]
    id2label = {0:"not offensive", 1:"offensive"}
    label2id = {"not offensive": 0, "offensive": 1}

    dataset = load_dataset("coltekin/offenseval2020_tr", cache_dir=args.cache_dir, trust_remote_code=True)
    dataset = dataset.rename_column("tweet", "text")
    dataset = dataset.rename_column("subtask_a", "label")

    dataset = dataset.remove_columns(["id"])

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
        })
    if args.model_id == "VRLLab/TurkishBERTweet":
        tokenized_dataset = dataset.map(lambda x: tokenizer(batch_preprocess2(x['text']), truncation=True, padding=True), batched=True).remove_columns(["text"])
    else:
        tokenized_dataset = dataset.map(preprocess_function, batched=True).remove_columns(["text"])

elif args.task == "nli":
    all_labels = [0,1,2]
    id2label = {0:"entailment", 1:"neutral", 2:"contradiction"}
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    dataset = load_dataset("facebook/xnli", "tr", cache_dir=args.cache_dir)
    if args.model_id == "VRLLab/TurkishBERTweet":
        tokenized_dataset = dataset.map(lambda x: tokenizer(preprocess(x['premise']), preprocess(x['hypothesis']), truncation=True, padding=True), batched=False)
    else:
        tokenized_dataset = dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], truncation=True, padding=True), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis"])

    test_df = dataset["test"].to_pandas()
else: # ner
    all_labels = [0,1,2,3,4,5,6,7,8]
    train_df = pd.read_csv("datasets/ner/train.csv", sep="\t", encoding="utf-8")
    val_df = pd.read_csv("datasets/ner/dev.csv", sep="\t", encoding="utf-8")
    test_df = pd.read_csv("datasets/ner/test.csv", sep="\t", encoding="utf-8")

    ner_labels = ['B-PERSON', 'O', 'I-ORGANIZATION', 'B-LOCATION', 'I-PERSON', 'I-LOCATION', 'B-ORGANIZATION', 'B-MISC', 'I-MISC']
    id2label = {i: label for i, label in enumerate(ner_labels)}
    label2id = {v: k for k, v in id2label.items()}
    
    for column in train_df.columns:
        train_df[column] = train_df[column].apply(ast.literal_eval)
        val_df[column] = val_df[column].apply(ast.literal_eval)
        test_df[column] = test_df[column].apply(ast.literal_eval)
    

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
        })

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

try:
    dataset = dataset.remove_columns(["__index_level_0__"])
except: 
    print("index column does not exist in the dataset!")


def init_model(config=None):

    # initialize a new wandb run
    with wandb.init(config=config, project=wandb_proj_name) as run:

        # if called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        run_name = f"task:{args.task}|model_id:{model_name}|batch:{config.batch_size}|lr:{config.learning_rate}"
        params = {
            "batch_size": config.batch_size, 
            "lr": config.learning_rate, 
            "run_name" : run_name
            }
        
        run.name = run_name
        train(params)
        
        torch.cuda.empty_cache()
        time.sleep(2)

def train(params):
    global sweep_no
    sweep_no += 1
    if args.task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir
        ).to("cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir
        ).to("cuda")

    training_args = TrainingArguments(
        output_dir=output_dir + model_name + "_" + str(sweep_no),
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=params["lr"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=2,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.2,
        eval_steps=0.1,
        report_to="wandb",
        run_name=params["run_name"],
        load_best_model_at_end=True,
        seed=42,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

    if args.task == "ner":
        predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        metrics = metric.compute(predictions=true_predictions, references=true_labels)

        result_dict["metrics"].append(metrics)
        result_dict["run name"].append(params["run_name"])
        result_dict["f1-macro"].append(metrics["overall_f1"])
        result_dict["confusion matrix"].append("")
    else:
        evaluate(trainer.model, params["run_name"])

def evaluate(model, run_name):
    global result_dict
    global test_df

    references = test_df["label"].to_list()
    predictions = []

    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        if args.model_id == "VRLLab/TurkishBERTweet":
            if args.task == "nli":
                inputs = tokenizer(preprocess(row['premise']), preprocess(row['hypothesis']), return_tensors="pt", truncation=True, max_length=512).to(model.device)
            else:
                inputs = tokenizer(preprocess(row["text"]), return_tensors="pt", truncation=True, max_length=512).to(model.device)
        else:
            if args.task == "nli":
                inputs = tokenizer(row['premise'], row['hypothesis'], return_tensors="pt", truncation=True, max_length=512).to(model.device)
            else:
                inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        y_pred = logits.argmax().item()
        predictions.append(y_pred)

    print(classification_report(references, predictions, labels=all_labels))

    conf_matrix = str(confusion_matrix(references, predictions).tolist())
    metrics = {
        "f1-macro" : f1_score(references, predictions, average='macro'),
        "f1-micro" : f1_score(references, predictions, average='micro'),
        "f1-weighted" : f1_score(references, predictions, average='weighted'),
        "precision" : precision_score(references, predictions, average='weighted'),
        "recall" : recall_score(references, predictions, average='weighted'),
        "accuracy" : accuracy_score(references, predictions)
    }

    result_dict["confusion matrix"].append(conf_matrix)
    result_dict["metrics"].append(metrics)
    result_dict["run name"].append(run_name)
    result_dict["f1-macro"].append(metrics["f1-macro"])


result_dict = {"run name":[], "metrics": [], "confusion matrix": [], "f1-macro":[]}
sweep_no = 0

if skip_train:
    for i in range(1,9):
        model = AutoModelForSequenceClassification.from_pretrained(output_dir + model_name + "_" + str(i), num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir)
        evaluate(model, run_name=None)

    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df)
    
    result_df.to_csv( file_name + ".csv", sep="\t", encoding="utf-8",  mode='a', header=False)

else:
    sweep_config = {
        'method': 'grid', 
        'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values': [2e-5, 5e-5, 1e-4]
                },
            'batch_size': {
                'values': [8, 16, 32]
                },
        }
    }

    # wandb.login(key=bb1d16d7e0ec95e5bf337e52d876ed48c1d773df)
    sweep_id = wandb.sweep(sweep_config, project=wandb_proj_name)

    wandb.agent(sweep_id, function=init_model, count=9)
    
    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df)
    
    result_df.to_csv( file_name + ".csv", sep="\t", encoding="utf-8",  mode='a', header=False)

    
