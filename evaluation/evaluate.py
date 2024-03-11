import argparse
import logging

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_path', type=str, required=True,
                        help="Path to tsv file with predicted labels. Predicted labels should be stored in the"
                             "'prediction' column as binary labels: 0 or 1.")
    parser.add_argument('--gold_labels_path', type=int, required=True,
                        help="Path to tsv file with ground truth labels. Ground truth labels should be stored in the"
                             "'correct' column as 'False' and 'True' strings.")

    args = parser.parse_args()
    return args


def main(args):
    predictions_path = args.predictions_path
    gold_labels_path = args.gold_labels_path

    predictions_df = pd.read_csv(predictions_path, sep='\t')
    test_df = pd.read_csv(gold_labels_path, sep='\t')
    test_df["label"] = test_df["correct"].astype(np.float32)

    pred_labels = predictions_df["prediction"].astype(np.int32).values
    if "prediction" not in predictions_df.columns:
        raise RuntimeError('prediction column is not found in submission file')
    predictions_unique_values = set((int(x) for x in predictions_df["prediction"].unique()))
    assert len(predictions_unique_values.intersection({0, 1})) == 2

    true_labels = test_df["label"].astype(np.int32).values

    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    print("Test\n")
    print(f"\tPublic precision: {p}\n")
    print(f"\tPublic recall: {r}\n")
    print(f"\tPublic F1: {f1}\n")
    print(f"\tPublic accuracy: {acc}\n")


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
