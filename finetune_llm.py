import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForTokenClassification
from trl import setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments, pipeline
import wandb
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import argparse
import evaluate
import ast
login(token="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")

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='meta-llama/Llama-3.1-8B-Instruct', type=str, help='model id')
parser.add_argument('--task', default='ner', type=str, help='sentiment/offensive/nli/ner')
parser.add_argument('--dataset', default='ner', type=str, help='tsad/offenseval2020_tr/xnli/ner')
parser.add_argument('--cache_dir', default='/mnt/proj1/dd-23-122/.cache', type=str, help='model cache dir')
parser.add_argument('--skip_train', default=True, type=bool, help='skip to inference')
parser.add_argument('--wandb_proj_name', default="bertweet", type=str, help='wandb project name')
args = parser.parse_args()


model_id = args.model_id
model_name = model_id.split("/")[-1]
cache_dir = args.cache_dir
skip_train = args.skip_train
output_dir = "outputs/" + args.task + "/" 
wandb_proj_name = args.wandb_proj_name
file_name = f"{args.task}_{model_name}_{args.dataset}"

f1 = evaluate.load("f1")
metric = evaluate.load("seqeval")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir)
tokenizer.padding_side = 'right' # to prevent warnings
tokenizer.pad_token = tokenizer.eos_token
#model.pad_token_id = tokenizer.pad_token_id
#model.config.pad_token_id = tokenizer.pad_token_id

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding="max_length", max_length=256, is_split_into_words=True)
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

def replace_label(sample):
    sample["label"] = ""
    return sample

def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.001, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    #outputs = pipe(prompt, num_beams=5, max_new_tokens=50, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    return predicted_answer.lower(), sample["messages"][2]["content"].lower()

def create_conversation(sample):
    if args.dataset == "xnli":
        text = f"""Premise: {sample["premise"]}
Hypothesis: {sample["hypothesis"]}
"""
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text},
                {"role": "assistant", "content": id2label[sample["label"]]}
                ]
            }
    else:
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": str(sample["text"])},
                {"role": "assistant", "content": str(sample["label"])}
                ]
            }


if args.task == "sentiment":
    all_labels = [0,1,2]
    system_message = """You are an AI agent. You are asked to classify the given text as 'neutral', 'positive or 'negative'."""
    label_dict = {0:"neutral", 1:"positive", 2:"negative"}
    id2label = {0:"neutral", 1:"positive", 2:"negative",}
    label2id = {"neutral": 0, "positive": 1, "negative": 2, "notr": 0, "positive": 1, "negative": 2}

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

elif args.task == "offensive":
    all_labels = [0,1]
    system_message = """You are an AI agent. You are asked to classify the given text as 'not offensive' or 'offensive'."""
    label_dict = {0:"not offensive", 1:"offensive"}
    id2label = {0:"not offensive", 1:"offensive"}
    label2id = {"not offensive": 0, "offensive": 1}

    dataset = load_dataset("coltekin/offenseval2020_tr", cache_dir=args.cache_dir, trust_remote_code=True)
    dataset = dataset.rename_column("tweet", "text")
    dataset = dataset.rename_column("subtask_a", "label")

    dataset = dataset.remove_columns(["id"])

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    train_df['label'] = train_df['label'].replace(label_dict)
    val_df['label'] = val_df['label'].replace(label_dict)
    test_df['label'] = test_df['label'].replace(label_dict)
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
        })

elif args.task == "nli":
    system_message = """You are an AI agent. You are asked to determining whether a 'hypothesis' is true (entailment), false (contradiction), or undetermined (neutral) given a 'premise'."""
    all_labels = [0,1,2]
    id2label = {0:"entailment", 1:"neutral", 2:"contradiction"}
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2, "true":0,"false":2, "undetermined":1}

    dataset = load_dataset("facebook/xnli", "tr", cache_dir=args.cache_dir)

else: # ner
    system_message = """You are an AI agent. You are asked to classify the following tokens for the named entity recognition tasks. The candidate labels are as follows:
    {'B-PERSON', 'O', 'I-ORGANIZATION', 'B-LOCATION', 'I-PERSON', 'I-LOCATION', 'B-ORGANIZATION', 'B-MISC', 'I-MISC'}."""

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
    dataset = dataset.rename_column("tokens", "text")
    dataset = dataset.rename_column("ner_tags", "label")
    """
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    """


if skip_train:
    print(f"INFERENCE MODE: {args.dataset}, MODEL_DIR: {output_dir}")
    if args.task == "nli":
        dataset["test"] = dataset["test"].map(create_conversation, remove_columns=["hypothesis","premise", "label"],batched=False)
    else:
        dataset["test"] = dataset["test"].map(create_conversation, remove_columns=["text", "label"],batched=False)
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir + model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        #low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    #merged_model.push_to_hub(hub_id)
    #tokenizer.push_to_hub(hub_id)

    #if args.task == "ner":
    #    pipe = pipeline("ner", model=merged_model, tokenizer=tokenizer, device="cuda")
    #else:
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, device="cuda")

else:
    if args.task == "nli":
        dataset = dataset.map(create_conversation, remove_columns=["hypothesis","premise", "label"],batched=False)
    #elif args.task != "ner":
    dataset = dataset.map(create_conversation, remove_columns=["text", "label"],batched=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    """
    if args.task == "ner":
        # Load model and tokenizer
        model = LlamaForTokenClassification.from_pretrained(
            model_id,
            device_map="auto",
            cache_dir=args.cache_dir,
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
        )
    else:
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
    )
    model.config.use_cache = False
    # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
        task_type="CAUSAL_LM",
    )


    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model_name = model_id.split("/")[-1]
    exp_name = f"task:{args.task}|dataset:{args.dataset}|model_id:{model_name}"
    wandb.init(project=wandb_proj_name, name=exp_name)

    train_args = TrainingArguments(
        output_dir=output_dir + model_name, # directory to save and repository id
        num_train_epochs=2,                     # number of training epochs
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=0.2,
        eval_steps=0.1,
        report_to="wandb",
        run_name=exp_name,
        load_best_model_at_end=True,
        seed=42,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        bf16=True,
        fp16=False,
    )

    max_seq_length = 1024 # max sequence length for model and packing of the dataset
    
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        #dataset_kwargs={
        #    "add_special_tokens": False,  # We template with special tokens
        #    "append_concat_token": False, # No need to add additional separator token
        #}
    )

    trainer.train()
    trainer.save_model()
    print(f"INFERENCE MODE: {args.dataset}, MODEL_DIR: {output_dir}")
    pipe = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer, device="cuda")

if args.task != "ner":
    predictions = []
    targets = []
    number_of_eval_samples = len(dataset["test"])
    # iterate over eval dataset and predict
    for s in tqdm(dataset["test"].shuffle().select(range(number_of_eval_samples))):
        pred, label = evaluate(s)

        try:
            predictions.append(label2id[pred])
        except:
            if "entailment" in pred.lower():
                pred = "entailment"
                predictions.append(label2id[pred])
            elif "contradiction" in pred.lower():
                pred = "contradiction"
                predictions.append(label2id[pred])
            elif "neutral" in pred.lower() or "undetermined":
                pred = "neutral"
                predictions.append(label2id[pred])
            else:
                print(f"Wrong prediction: {pred}")
                continue
        targets.append(label2id[label])
        
    metrics = {
        "f1-macro" : f1_score(targets, predictions, average='macro'),
        "f1-micro" : f1_score(targets, predictions, average='micro'),
        "f1-weighted" : f1_score(targets, predictions, average='weighted'),
        "precision" : precision_score(targets, predictions, average='weighted'),
        "recall" : recall_score(targets, predictions, average='weighted'),
        "accuracy" : accuracy_score(targets, predictions)
    }
    print(metrics)
else:
    predictions = []
    targets = []
    texts = []
    number_of_eval_samples = len(dataset["test"])
    # iterate over eval dataset and predict
    for s in tqdm(dataset["test"].shuffle().select(range(number_of_eval_samples))):
        try:
            pred, label = evaluate(s) 
            #pred = ast.literal_eval(pred)
            predictions.append(pred)
            targets.append(label)
            texts.append(str(s["messages"][1]))
        except:
            print(f"""Exception!!!\n{str(s["messages"][1])}""")

    df = pd.DataFrame({"text": texts,"labels": targets, "predictions":predictions})
    df.to_csv("ner_predictions.csv", encoding="utf-8", sep="\t")