import argparse
import logging
import os.path

import pandas as pd


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_path', type=str, required=False,
                        default="../data/parquet/train.parquet")
    parser.add_argument('--input_test_path', type=str, required=False,
                        default="../data/parquet/test.parquet")
    parser.add_argument('--output_train_path', type=str, required=False,
                        default="../data/tsv/train.tsv")
    parser.add_argument('--output_test_path', type=str, required=False,
                        default="../data/tsv/dev.tsv")

    args = parser.parse_args()
    return args


def main(args):
    input_train_path = args.input_train_path
    input_test_path = args.input_test_path
    output_train_path = args.output_train_path
    output_test_path = args.output_test_path
    output_train_dir = os.path.dirname(output_train_path)
    output_test_dir = os.path.dirname(output_test_path)
    if not os.path.exists(output_train_dir) and output_train_dir != '':
        os.makedirs(output_train_dir)
    if not os.path.exists(output_test_dir) and output_test_dir != '':
        os.makedirs(output_test_dir)

    train_df = pd.read_parquet(input_train_path)
    test_df = pd.read_parquet(input_test_path)

    train_df["sample_id"] = train_df.index
    test_df["sample_id"] = test_df.index

    train_df.drop(['id', 'complexityType'], axis=1, inplace=True)
    test_df.drop(['id', 'complexityType'], axis=1, inplace=True)

    train_df[["sample_id", "question", "answerEntity", "questionEntity",
              "groundTruthAnswerEntity", "graph", "correct"]].to_csv(output_train_path,
                                                                     index=False, sep='\t')
    test_df[["sample_id", "question", "answerEntity", "questionEntity",
             "groundTruthAnswerEntity", "graph", "correct"]].to_csv(output_test_path,
                                                                    index=False, sep='\t')


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
