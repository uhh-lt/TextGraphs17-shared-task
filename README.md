# TextGraphs17-shared-task


## Data

We present a dataset for graph-based question answering. The dataset consists of <question; candidate answer> pairs. For each candidate, we present a graph that is obtained by finding the shortest path between named entities mentioned in a question and a candidate answer. As a knowledge graph, we adopted Wikidata. Thus, answer candidate corresponds to a node from the Wikidata knowledge graph. 

The data format used in our dataset in compatible with NetworkX. For an example on how to work with shortest path graph, please see [this script](https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/question_graph_examples).

All the data will be available [here](https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/data/tsv).

For now, we released [train+dev](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/data/tsv/train_dev.tsv) split. 



## Data vizualization

For your convenience, we provide a [visualization script](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/visualization/draw_random_question_graphs.py) question-candidate graphs to help you better understand the graph structures. Pre-vizualized graphs for all candidate answers of 10 random questions are available [here](https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/question_graph_examples).  

https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/question_graph_examples


## Baselines

We propose two BERT-based [baselines](https://github.com/uhh-lt/TextGraphs17-shared-task/baselines/bert_baselines.ipynb):

* A BERT-based binary classifier which does not take graphs into account.

* A BERT-based binary classifier which trains on concatenated samples consisting of: (i) question, (ii) linearized graph of a candidate answer.

## Evaluation

We see the task as a binary classification. Given a question, answer candidate, and a subgraph from knowledge grpah, your goal is assign a binary label to each pair.

For evaluation, we will adopt precision, recall, F1-measure, and accuracy. For your convenience, we publish an [evaluation script](https://github.com/uhh-lt/TextGraphs17-shared-task/blob/main/evaluation/evaluate.py).


