import argparse
import json
import logging
import os
import random
import re
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch.cuda
from matplotlib import pyplot as plt
from networkx import spring_layout
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def split_node_labels(n_label, max_line_length=16, min_line_length=4):
    label_length = len(n_label)
    accum_length = 0
    lines = []
    words = n_label.split()
    curr_s = f"{words[0]}"
    for w in words[1:]:
        curr_len = len(curr_s)
        w_len = len(w)
        if curr_len + w_len > max_line_length and label_length - accum_length > min_line_length:
            lines.append(curr_s)
            curr_s = w
            accum_length += len(curr_s)
        else:
            curr_s += f" {w}"
    if len(curr_s) > 0:
        lines.append(curr_s)

    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tsv', type=str,
                        default="../data/tsv/train.tsv")
    parser.add_argument('--num_questions', type=int, default=10, required=False)
    parser.add_argument('--output_dir', type=str,
                        default="../question_graph_examples/")

    args = parser.parse_args()

    return args


def main(args):
    input_tsv = args.input_tsv
    num_questions = args.num_questions
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df = pd.read_csv(input_tsv, sep='\t')
    qs = tuple(df["question"].unique())
    random_qs = random.sample(qs, k=num_questions)

    for i, q in enumerate(random_qs):
        ddf = df[df["question"] == q]
        for _, row in ddf.iterrows():
            true_flag = row["correct"]
            assert isinstance(true_flag, bool)
            assert row["question"] == q
            graph_json = eval(row["graph"])
            graph_json["directed"] = True

            sample_id = row["sample_id"]
            cand_s = row["answerEntity"].replace('/', '|').replace('\\', '|')
            true_s = row["groundTruthAnswerEntity"].replace('/', '|').replace('\\', '|')

            nx_graph = nx.node_link_graph(graph_json, )

            candidate_color = "green" if true_flag else "red"
            color_map = {"ANSWER_CANDIDATE_ENTITY": candidate_color,
                         "QUESTIONS_ENTITY": '#2A4CC6'}
            node_colors = {}
            labels = {}
            for n_dict in graph_json["nodes"]:
                n_id = n_dict["id"]
                n_label = n_dict["label"]
                n_label = "None" if n_label is None else n_label
                n_type = n_dict["type"]
                n_label = split_node_labels(n_label, max_line_length=13, min_line_length=4)
                labels[n_id] = n_label
                color = color_map[n_type] if color_map.get(n_type) is not None else "#808080"
                node_colors[n_id] = color

            node_colors_np = [node_colors[key] for key in sorted(node_colors.keys())]
            node_colors_np = np.array(node_colors_np)
            edge_labels = {}
            for e_dict in graph_json["links"]:
                src_i = e_dict["source"]
                trg_i = e_dict["target"]
                e_label = e_dict["label"]
                e_label = split_node_labels(e_label, max_line_length=12, min_line_length=3)
                edge_labels[(src_i, trg_i)] = e_label

            plt.title(split_node_labels(q, max_line_length=64, min_line_length=16), fontsize=12)
            # pos = nx.spring_layout(nx_graph)
            try:
                pos = nx.planar_layout(nx_graph, scale=0.05)
            except Exception as e:
                pos = nx.spring_layout(nx_graph, scale=0.1)
            nx.draw(nx_graph, pos=pos, node_size=250, alpha=0.8, node_color=node_colors_np, font_size=12,
                    font_weight='bold')

            pos_edge_labels = {}
            y_off = 0.05
            max_pos = max(v[1] for v in pos.values())
            min_pos = min(v[1] for v in pos.values())
            delta_pos = max_pos - min_pos
            for k, v in pos.items():
                pos_edge_labels[k] = (v[0], v[1] + y_off * delta_pos)

            nx.draw_networkx_edge_labels(
                nx_graph, pos_edge_labels,
                edge_labels=edge_labels,
                font_color='red',
                label_pos=0.375,
                font_size=6
            )
            pos_node_labels = {}
            y_off = 0.075  # offset on the y axis
            max_pos = max(v[1] for v in pos.values())
            min_pos = min(v[1] for v in pos.values())
            delta_pos = max_pos - min_pos

            for k, v in pos.items():
                offset = y_off * delta_pos

                pos_node_labels[k] = (v[0], v[1] - offset)
            nx.draw_networkx_labels(nx_graph, pos_node_labels, labels, font_size=8)

            fname = f"{true_s}_{sample_id}_{cand_s}.png"
            output_subdir = os.path.join(output_dir, f"question_{i}/")
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            x0, x1 = plt.xlim()
            y0, y1 = plt.ylim()
            plt.xlim(x0 * 1.1, x1 * 1.1)
            plt.ylim(y0 * 1.1, y1 * 1.1)

            output_graph_path = os.path.join(output_dir, f"question_{i}/", fname)
            plt.savefig(output_graph_path, format="PNG")
            plt.clf()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    args = parse_args()
    main(args)
