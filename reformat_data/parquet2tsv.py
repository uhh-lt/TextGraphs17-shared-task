import argparse
import logging
import os.path
import random
from typing import Dict
import pandas as pd

NONE_WIKIDATA_IDS = {
    "Q26189069": "Wikimedia template",
    "Q25707836": "NONE",
    "Q9139189": "book by Eliza Orzeszkowa",
    "Q1321674": "NONE",
    "Q25405526": "NONE",
    "Q63992256": "Model: United States",
    "Q7025337": "NONE",
    "Q27131943": "NONE"
}


def create_wikidata_id2name_map(graphs):
    wikidata_id2name: Dict[str, str] = {}
    for question_graph in graphs:
        for node_dict in question_graph["nodes"]:
            wikidata_id = node_dict["name_"]
            wikidata_name = node_dict["label"]
            wikidata_id2name[wikidata_id] = wikidata_name
    return wikidata_id2name


def wikidata_ids2names(ids_str, wikidata_id2name, sep='\\'):
    ids_list = ids_str.split(',')
    names = [
        wikidata_id2name[w_id.strip()] if wikidata_id2name.get(w_id.strip()) is not None else NONE_WIKIDATA_IDS.get(
            w_id.strip(), "NONE")
        for w_id in ids_list]
    # names = [wikidata_id2name[w_id] for w_id in ids_list if wikidata_id2name[w_id]
    # is not None else NONE_WIKIDATA_IDS[w_id]]
    for name in names:
        assert sep not in name
    s = sep.join(names)

    return s


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_path', type=str, required=False,
                        default="../data/parquet/train-t5-xl.parquet")
    parser.add_argument('--input_dev_path', type=str, required=False,
                        default="../data/parquet/test-t5-xl.parquet")
    parser.add_argument('--num_debug_test_questions', type=int, default=1000)
    parser.add_argument('--output_train_path', type=str, required=False,
                        default="../data/tsv/train.tsv")
    parser.add_argument('--output_dev_path', type=str, required=False,
                        default="../data/tsv/dev.tsv")
    parser.add_argument('--output_test_path', type=str, required=False,
                        default="../data/tsv/test.tsv")

    args = parser.parse_args()
    return args


def main(args):
    input_train_path = args.input_train_path
    input_dev_path = args.input_dev_path
    num_debug_test_questions = args.num_debug_test_questions
    output_train_path = args.output_train_path
    output_dev_path = args.output_dev_path
    output_test_path = args.output_test_path
    output_train_dir = os.path.dirname(output_train_path)
    output_dev_dir = os.path.dirname(output_dev_path)
    output_test_dir = os.path.dirname(output_test_path)
    if not os.path.exists(output_train_dir) and output_train_dir != '':
        os.makedirs(output_train_dir)
    if not os.path.exists(output_dev_dir) and output_dev_dir != '':
        os.makedirs(output_dev_dir)
    if not os.path.exists(output_test_dir) and output_test_dir != '':
        os.makedirs(output_test_dir)
    input_train_dir = os.path.dirname(input_train_path)
    input_dev_dir = os.path.dirname(input_dev_path)
    train_unprocessed_tsv_path = os.path.join(input_train_dir, "train_unprocessed.tsv")
    dev_unprocessed_tsv_path = os.path.join(input_dev_dir, "dev_unprocessed.tsv")

    train_df = pd.read_parquet(input_train_path)
    dev_df = pd.read_parquet(input_dev_path)

    train_df.to_csv(train_unprocessed_tsv_path, index=False, sep='\t')
    dev_df.to_csv(dev_unprocessed_tsv_path, index=False, sep='\t')
    logging.info(f"Loaded train (size: {train_df.shape})")
    logging.info(f"Loaded dev (size: {dev_df.shape})")
    unique_train_questions = train_df["question"].unique()
    unique_dev_questions = dev_df["question"].unique()
    unique_train_question_ids = train_df["id"].unique()
    unique_dev_question_ids = dev_df["id"].unique()
    assert len(unique_train_questions) == len(unique_train_question_ids)
    assert len(unique_dev_questions) == len(unique_dev_question_ids)
    logging.info(f"There are {len(unique_train_questions)} unique questions in train")
    logging.info(f"There are {len(unique_dev_questions)} unique questions in dev")

    train_df["bad_candidate"] = train_df["answerEntity"].apply(lambda x: True if x.startswith('P') else False)
    train_df = train_df[~train_df["bad_candidate"]]
    dev_df["bad_candidate"] = dev_df["answerEntity"].apply(lambda x: True if x.startswith('P') else False)

    dev_df = dev_df[~dev_df["bad_candidate"]]
    logging.info(f"Train size after bad candidate deletion: {train_df.shape}")
    logging.info(f"Dev size after bad candidate deletion: {dev_df.shape}")

    train_df.dropna(subset=('questionEntity',), inplace=True)
    dev_df.dropna(subset=('questionEntity',), inplace=True)
    logging.info(f"Train size after no-entity questions drops: {train_df.shape}")
    logging.info(f"Dev size after no-entity questions drops: {dev_df.shape}")

    train_df["graph"] = train_df["graph"].apply(eval)
    dev_df["graph"] = dev_df["graph"].apply(eval)
    wikidata_id2name = create_wikidata_id2name_map(train_df["graph"].values)
    wikidata_id2name.update(create_wikidata_id2name_map(dev_df["graph"].values))
    for i, (k, v) in enumerate(wikidata_id2name.items()):
        if i > 5:
            break
        print(k, '|', v)
    for val in train_df["graph"]:
        del val["graph"]
        del val["directed"]
        del val["multigraph"]
    for val in dev_df["graph"]:
        del val["graph"]
        del val["directed"]
        del val["multigraph"]

    train_df["answerEntityId"] = train_df["answerEntity"]
    train_df["questionEntityId"] = train_df["questionEntity"]
    train_df["groundTruthAnswerEntityId"] = train_df["groundTruthAnswerEntity"]
    dev_df["answerEntityId"] = dev_df["answerEntity"]
    dev_df["questionEntityId"] = dev_df["questionEntity"]
    dev_df["groundTruthAnswerEntityId"] = dev_df["groundTruthAnswerEntity"]

    logging.info("Mapping train answerEntity wikidata ids to labels")
    train_df["answerEntity"] = train_df["answerEntity"].apply(lambda x: wikidata_ids2names(x, wikidata_id2name))
    logging.info("Mapping train questionEntity wikidata ids to labels")
    train_df["questionEntity"] = train_df["questionEntity"].apply(lambda x: wikidata_ids2names(x, wikidata_id2name))
    logging.info("Mapping train groundTruthAnswerEntity wikidata ids to labels")
    train_df["groundTruthAnswerEntity"] = train_df["groundTruthAnswerEntity"].apply(
        lambda x: wikidata_ids2names(x, wikidata_id2name))

    logging.info("Mapping dev answerEntity wikidata ids to labels")
    dev_df["answerEntity"] = dev_df["answerEntity"].apply(lambda x: wikidata_ids2names(x, wikidata_id2name))
    logging.info("Mapping dev questionEntity wikidata ids to labels")
    dev_df["questionEntity"] = dev_df["questionEntity"].apply(lambda x: wikidata_ids2names(x, wikidata_id2name))
    logging.info("Mapping dev groundTruthAnswerEntity wikidata ids to labels")
    dev_df["groundTruthAnswerEntity"] = dev_df["groundTruthAnswerEntity"].apply(
        lambda x: wikidata_ids2names(x, wikidata_id2name))

    unique_dev_question_ids = list(dev_df["id"].unique())
    random.shuffle(unique_dev_question_ids)
    test_question_ids = unique_dev_question_ids[:num_debug_test_questions]
    dev_question_ids = unique_dev_question_ids[num_debug_test_questions:]
    num_public_test_questions = num_debug_test_questions // 2
    public_test_q_ids = set(test_question_ids[:num_public_test_questions])
    private_test_q_ids = set(test_question_ids[num_public_test_questions:])

    logging.info(f"Public test unique questions: {len(public_test_q_ids)}")
    logging.info(f"Private test unique questions: {len(private_test_q_ids)}")
    logging.info(f"Dev unique questions: {len(dev_question_ids)}")

    public_test_df = dev_df[dev_df["id"].isin(public_test_q_ids)]
    private_test_df = dev_df[dev_df["id"].isin(private_test_q_ids)]
    dev_df = dev_df[dev_df["id"].isin(dev_question_ids)]

    logging.info(f"Public test df size: {public_test_df.shape}")
    logging.info(f"Private test df size: {private_test_df.shape}")
    logging.info(f"Dev df size: {dev_df.shape}")

    private_test_df["subset"] = "private"
    public_test_df["subset"] = "public"
    test_df = pd.concat([public_test_df, private_test_df]).sample(frac=1)

    train_df.drop(['id', 'complexityType'], axis=1, inplace=True)
    dev_df.drop(['id', 'complexityType'], axis=1, inplace=True)

    train_df["sample_id"] = list(range(train_df.shape[0]))
    dev_df["sample_id"] = list(range(dev_df.shape[0]))
    test_df["sample_id"] = list(range(test_df.shape[0]))

    train_df[["sample_id", "question", "answerEntity", "questionEntity", "groundTruthAnswerEntity", "answerEntityId",
              "questionEntityId", "groundTruthAnswerEntityId", "graph", "correct"]].to_csv(output_train_path,
                                                                                           index=False, sep='\t')
    # test_df["prediction"] = test_df["correct"].astype(np.float32)
    # test_df["subset"] = "private"
    # test_df[["sample_id", "question", "answerEntity", "questionEntity",
    #          "groundTruthAnswerEntity", "graph", "correct", "prediction", "subset"]].to_csv(output_test_path,
    #                                                                 index=False, sep='\t')
    dev_df[["sample_id", "question", "answerEntity", "questionEntity", "groundTruthAnswerEntity", "answerEntityId",
            "questionEntityId", "groundTruthAnswerEntityId", "graph", "correct"]].to_csv(output_dev_path,
                                                                                         index=False, sep='\t')
    test_df[["sample_id", "subset", "question", "answerEntity", "questionEntity", "groundTruthAnswerEntity",
             "answerEntityId", "questionEntityId", "groundTruthAnswerEntityId", "graph", "correct"]] \
        .to_csv(output_test_path, index=False, sep='\t')


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
