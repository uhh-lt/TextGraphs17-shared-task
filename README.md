# TextGraphs17-shared-task


## Data

We present a dataset for graph-based question answering. The dataset consists of <question; candidate answer> pairs. For each candidate, we present a graph that is obtained by finding the shortest path between named entities mentioned in a question and a candidate answer. As a knowledge graph, we adopted Wikidata. Our dataset has the following fields:

* `sample_id` - an identifier for <question, candidate answer>;

* `question` - question text;

* `questionEntity` - comma-separated list of names (textual strings) for Wikidata concepts mentioned in a given question;

* `answerEntity` - a textual name of candidate answer (candidate is a concept from Wikidata) for the given question;

* `groundTruthAnswerEntity` - a textual name of ground truth answer (answer is a concept from Wikidata) for the given question;

* `answerEntityId` - a Wikidata id of candidate answer (see "answerEntity" column). Example: "Q2599";

* `questionEntityId` - a comma-separated list of Wikidata ids for concepts mentioned in a given question (list of ids for mentions from "questionEntity" column);

* `groundTruthAnswerEntityId` - a Wikidata id of ground truth answer (see "answerEntity" column). Example: "Q148234";

* `correct` - either "True" or "False". The field indicates whether a <question, answer candidate> is correct, i.e., candidate answer is a true answer to the given question;

* `graph` - a shortest-path graph for a given <question, candidate answer> pair. The graph is obtained by taking the shortest paths from all mentioned concepts ("questionEntityId" column) to a candidate answer("answerEntityId" column) in the knowledge graph of Wikidata. The graph is stored in "node-link" JSON format from NetworkX. You can import the graph using the [node_link_graph](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_graph.html).


For an example on how to draw shortest path graphs using NetworkX, please see the "Data vizualization" section. You can use a Wikidata id to access  Wikidata concepts. E.g., [Q148234](https://www.wikidata.org/wiki/Q148234) or [Q51056](https://www.wikidata.org/wiki/Q51056).


Train dataset is available [here](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/data/tsv/train.tsv). It can be used for initial experiments. Our dataset is also available at [HuggingFace](https://huggingface.co/datasets/s-nlp/TextGraphs17-shared-task-dataset).


## Data vizualization

For your convenience, we provide a [visualization script](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/visualization/draw_random_question_graphs.py) question-candidate graphs to help you better understand the graph structures. You can run the visualization script as follows:

```
python visualization/draw_random_question_graphs.py --input_tsv "data/tsv/train.tsv" \
--num_questions 10 \
--output_dir "question_graph_examples/"
```

Pre-vizualized graphs for all candidate answers of 10 random questions are available [here](https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/question_graph_examples).

## Baselines

We propose two BERT-based [baselines](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/baselines/bert_baselines.ipynb):

* A BERT-based binary classifier which does not take graphs into account.

* A BERT-based binary classifier which trains on concatenated samples consisting of: (i) question, (ii) linearized graph of a candidate answer.

## Evaluation

We see the task as a binary classification. Given a question, answer candidate, and a subgraph from knowledge grpah, your goal is assign a binary label to each pair.

For evaluation, we will adopt precision, recall, F1-measure, and accuracy. For your convenience, we publish an [evaluation script](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/evaluation/evaluate.py). The evaluation script can be executed as follows:
```
python visualization/draw_random_question_graphs.py \
--predictions_path <path to tsv-file with predicted labels> \
--gold_labels_path "data/tsv/train.tsv"
```


