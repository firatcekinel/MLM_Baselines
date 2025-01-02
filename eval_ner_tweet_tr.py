import pandas as pd
import csv
import ast
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
from Preprocessor import preprocess, batch_preprocess, batch_preprocess2
import torch

model_id = "VRLLab/TurkishBERTweet"
#model_id = "dbmdz/bert-base-turkish-cased"
#model_id = "FacebookAI/xlm-roberta-large"


cache_dir = "/mnt/proj1/dd-23-122/.cache"
model_name = model_id.split("/")[-1]
output_dir = "outputs/ner/" 

metric = evaluate.load("seqeval")
result_dict = {"metrics": [], "model_id": [], "f1-macro":[]}

ner_labels = ['B-PERSON', 'O', 'I-ORGANIZATION', 'B-LOCATION', 'I-PERSON', 'I-LOCATION', 'B-ORGANIZATION', 'B-MISC', 'I-MISC']
id2label = {i: label for i, label in enumerate(ner_labels)}
label2id = {v: k for k, v in id2label.items()}
all_labels = [0,1,2,3,4,5,6,7,8]

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def preprocess_function(examples):
    if model_id == 'VRLLab/TurkishBERTweet':
        return tokenizer(preprocess(examples["text"]), truncation=True)   
    return tokenizer(examples["text"], truncation=True)

def tokenize_and_align_labels(examples):
    if model_id == 'VRLLab/TurkishBERTweet':
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


test_df = pd.read_csv("datasets/ner/twitter_tr_dataset_bio.csv", sep="\t", encoding="utf-8")
for column in test_df.columns:
    test_df[column] = test_df[column].apply(ast.literal_eval)


dataset = DatasetDict({
    "test": Dataset.from_pandas(test_df),
    })

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["test"].column_names,
)



for i in range(1,9):
    true_labels_list = []
    true_predictions_list = []
    model = AutoModelForTokenClassification.from_pretrained(output_dir + model_name + "_" + str(i), num_labels=len(all_labels), id2label=id2label, label2id=label2id, cache_dir=cache_dir)
    for row in tokenized_dataset["test"]:
        input_ids = torch.tensor([row["input_ids"]])
        attention_mask = torch.tensor([row["attention_mask"]])
        labels = torch.tensor([row["labels"]])

        # Pass tensors to the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Convert logits to predictions
        predictions = logits.argmax(dim=-1).cpu().numpy()

        # Remove ignored index (-100) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels.numpy()]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels.numpy())
        ]
         # Collect all predictions and labels
        true_labels_list.append(true_labels)
        true_predictions_list.append(true_predictions)

    metrics = metric.compute(predictions=true_predictions_list, references=true_labels_list)

    result_dict["metrics"].append(metrics)
    result_dict["model_id"].append(model_id)
    result_dict["f1-macro"].append(metrics["overall_f1"])

print(result_dict)