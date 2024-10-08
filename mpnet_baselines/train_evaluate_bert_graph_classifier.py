import argparse
import json
import logging
import os
import random
import statistics
from typing import List

# metric = load_metric('accuracy')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    EarlyStoppingCallback
from transformers import AutoTokenizer


def linearize_graph(graph_dict, sep_token):
    nodes = sorted((node_dict for node_dict in graph_dict["nodes"]), key=lambda d: d["id"])
    for n_id, node_dict in enumerate(nodes):
        assert n_id == node_dict["id"]
    src_node_id2links = {}
    for link_dict in graph_dict["links"]:
        link_src = link_dict["source"]
        if src_node_id2links.get(link_src) is None:
            src_node_id2links[link_src] = []
        src_node_id2links[link_src].append(link_dict)
    # graph_s = ""
    link_list = []

    for n_id, node_dict in enumerate(nodes):
        links = src_node_id2links.get(n_id, list())
        start_label = node_dict["label"]
        if node_dict["type"] == "ANSWER_CANDIDATE_ENTITY":
            start_label = f"{sep_token} {start_label} {sep_token}"
        for link_dict in links:
            target_label = nodes[link_dict["target"]]["label"]
            if nodes[link_dict["target"]]["type"] == "ANSWER_CANDIDATE_ENTITY":
                target_label = f"{sep_token} {target_label} {sep_token}"
            link_s = f"{start_label}, {link_dict['label']}, {target_label}"
            link_list.append(link_s)
            # graph_s += link_s
    graph_s = ' ; '.join(link_list)

    return graph_s


def create_class_weights(df):
    class_counts = df["label"].value_counts()
    num_samples = df["label"].shape[0]
    # pos_weight = class_counts[0] / class_counts[1]
    class_weight_0 = class_counts[1] / num_samples
    class_weight_1 = class_counts[0] / num_samples

    class_weights = [class_weight_0, class_weight_1]
    return class_weights


class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        class_weights = kwargs.pop("class_weights")
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # self.class_weights = kwargs["class_weights"]
        self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = float(accuracy_score(labels, predictions, normalize=True, sample_weight=None, average="binary"))

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': p,
        'recall': r,
    }


class QuestionAnswerDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, text_graph_format,
                 debug_flag=False):
        super(QuestionAnswerDataset).__init__()
        assert text_graph_format in ("text_only", "graph_only", "text_graph")

        self.text_graph_format = text_graph_format
        self.labels = torch.tensor(df.label.values, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length
        self.debug_flag = debug_flag

        self.input_samples: List[str] = []
        for _, row in df.iterrows():
            input_sample = self.df_row2string_sample(row_dict=row)
            self.input_samples.append(input_sample)
        self.tokenized_input = self.tokenizer(self.input_samples, padding='max_length', truncation=True)
        assert len(self.tokenized_input["input_ids"]) == len(self.labels)
        if debug_flag:
            real_max_length = max(len(t) for t in self.tokenized_input["input_ids"])
            logging.info(f"Tokenized input examples ({self.text_graph_format}, {real_max_length}):")
            for i, inp_ids in enumerate(self.tokenized_input["input_ids"]):
                if i > 5:
                    break
                tokens = self.tokenizer.convert_ids_to_tokens(inp_ids)
                tok_s = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in tokens))
                logging.info(tok_s)

    def df_row2string_sample(self, row_dict):
        q = row_dict["question"]
        if self.text_graph_format == "text_only":
            answer_s = row_dict["answerEntity"]
            input_s = f"{q} {self.sep_token} {answer_s}"
        elif self.text_graph_format == "graph_only":
            ling = row_dict["linearized_graph"]
            input_s = ling
        elif self.text_graph_format == "text_graph":
            ling = row_dict["linearized_graph"]
            input_s = f"{q} {self.sep_token} {ling}"
        else:
            raise RuntimeError(f"Unsupported text_graph_format: {self.text_graph_format}")

        return input_s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_input["input_ids"][idx],
            "attention_mask": self.tokenized_input["attention_mask"][idx],  # [0],
            "labels": self.labels[idx]}


def extract_candidate_name_from_graph(graph_dict):
    for node_dict in graph_dict["nodes"]:
        if node_dict["type"] == "ANSWER_CANDIDATE_ENTITY":
            return node_dict["label"]
    raise RuntimeError(f"Found no candidate name for graph: {graph_dict}")


def main(args):
    input_data_dir = args.input_data_dir
    max_length = args.max_length
    batch_size = args.batch_size
    text_graph_format = args.text_graph_format
    gradient_accumulation_steps = args.gradient_accumulation_steps
    debug_flag = args.debug
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    base_model_name = args.base_model_name
    fp16 = args.fp16
    output_dir = args.output_dir

    output_finetuned_dir = os.path.join(output_dir, f"finetuned_models_{text_graph_format}/")
    if not os.path.exists(output_finetuned_dir):
        os.makedirs(output_finetuned_dir)

    input_train_path = os.path.join(input_data_dir, "train.tsv")
    input_dev_path = os.path.join(input_data_dir, "dev.tsv")
    large_test_path = os.path.join(input_data_dir, "large_test.tsv")
    small_test_path = os.path.join(input_data_dir, "small_test.tsv")

    train_df = pd.read_csv(input_train_path, sep='\t')
    dev_df = pd.read_csv(input_dev_path, sep='\t')
    large_test_df = pd.read_csv(large_test_path, sep='\t')
    small_test_df = pd.read_csv(small_test_path, sep='\t')


    train_df["label"] = train_df["correct"].astype(np.float32)
    dev_df["label"] = dev_df["correct"].astype(np.float32)
    large_test_df["label"] = large_test_df["correct"].astype(np.float32)
    small_test_df["label"] = small_test_df["correct"].astype(np.float32)
    if debug_flag:
        logging.info(f"Unique labels: {set(train_df['label'].unique())} {set(small_test_df['label'].unique())}")

    train_df["graph"] = train_df["graph"].apply(eval)
    dev_df["graph"] = dev_df["graph"].apply(eval)
    large_test_df["graph"] = large_test_df["graph"].apply(eval)
    small_test_df["graph"] = small_test_df["graph"].apply(eval)

    for df in (train_df, dev_df, large_test_df):
        df.rename(columns={
            "answerEntity": "answerEntityId",
            "questionEntity": "questionEntityId",
            "groundTruthAnswerEntity": "groundTruthAnswerEntityId"
        }, inplace=True)
        df["answerEntity"] = df["graph"].apply(extract_candidate_name_from_graph)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    sep_token = tokenizer.sep_token
    if text_graph_format in ("graph_only", "text_graph"):
        train_df["linearized_graph"] = train_df["graph"].apply(lambda x: linearize_graph(x, sep_token))
        dev_df["linearized_graph"] = dev_df["graph"].apply(lambda x: linearize_graph(x, sep_token))
        large_test_df["linearized_graph"] = large_test_df["graph"].apply(lambda x: linearize_graph(x, sep_token))
        small_test_df["linearized_graph"] = small_test_df["graph"].apply(lambda x: linearize_graph(x, sep_token))

    train_dataset = QuestionAnswerDataset(df=train_df, tokenizer=tokenizer, max_length=max_length,
                                          text_graph_format=text_graph_format, debug_flag=debug_flag)
    dev_dataset = QuestionAnswerDataset(df=dev_df, tokenizer=tokenizer, max_length=max_length,
                                        text_graph_format=text_graph_format, debug_flag=debug_flag)
    small_test_dataset = QuestionAnswerDataset(df=small_test_df, tokenizer=tokenizer, max_length=max_length,
                                               text_graph_format=text_graph_format, debug_flag=debug_flag)
    large_test_dataset = QuestionAnswerDataset(df=large_test_df, tokenizer=tokenizer, max_length=max_length,
                                               text_graph_format=text_graph_format, debug_flag=debug_flag)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)

    # model = AutoModelForSequenceClassification.from_pretrained(base_model_name,
    #                                                            num_labels=2)

    train_args = TrainingArguments(
        output_finetuned_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        fp16=fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=0.01,
        save_total_limit=2,
        seed=42,
        push_to_hub=False,
    )
    class_weights = create_class_weights(train_df)
    logging.info(f"Class weights: {class_weights}")

    trainer = WeightedTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), ],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,

    )

    logging.info("Training...")
    trainer.train()
    logging.info("Finished training. Evaluating....")
    dev_eval_dict = trainer.evaluate()
    logging.info(f"Dev:")
    print(f"Dev:")
    for k, v in dev_eval_dict.items():
        print(f"Final score (dev) {k}: {v}")
        logging.info(f"Final score (dev) {k}: {v}")

    large_test_eval_dict = trainer.evaluate(large_test_dataset)
    print(f"Large test:")
    logging.info(f"Large test:")
    for k, v in large_test_eval_dict.items():
        print(f"Final score (large test) {k}: {v}")
        logging.info(f"Final score (large test) {k}: {v}")

    small_test_eval_dict = trainer.evaluate(small_test_dataset)
    print(f"Small test:")
    logging.info(f"Small test:")
    for k, v in small_test_eval_dict.items():
        print(f"Final score (small test) {k}: {v}")
        logging.info(f"Final score (small test) {k}: {v}")

    logging.info("Finished Evaluation....")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_data_dir', type=str, required=True)
    # parser.add_argument('--base_model_name', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, required=False, default=16)
    # parser.add_argument('--max_length', type=int, required=False, default=512)
    # parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1)
    # parser.add_argument('--text_graph_format', type=str, required=False,
    #                     choices=("text_only", "graph_only", "text_graph"), default="graph_only")
    #
    # parser.add_argument('--num_epochs', type=int, required=False, default=50)
    # parser.add_argument('--warmup_ratio', type=float, required=False, default=0.1)
    # parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    # parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    # parser.add_argument("--fp16", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--input_data_dir', type=str, required=False,
                        default="/home/c204/University/NLP/KGQA_mintaka_data/TextGraphs17-shared-task/dataset_naacl_2025_resplit/")
    parser.add_argument('--base_model_name', type=str, required=False, default="prajjwal1/bert-tiny")
    parser.add_argument('--batch_size', type=int, required=False, default=3)
    parser.add_argument('--max_length', type=int, required=False, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1)
    parser.add_argument('--text_graph_format', type=str, required=False,
                        choices=("text_only", "graph_only", "text_graph"), default="text_graph")

    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    parser.add_argument('--warmup_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--warmup_steps', type=int, required=False, default=0)

    parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", default=True)
    parser.add_argument('--output_dir', type=str, required=False,
                        default="./DELETE/")

    args = parser.parse_args()
    main(args)
