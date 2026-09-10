# CSE 5525 Homework 1: Sentiment Classification

This repository contains implementations of several binary sentiment classifiers for CSE 5525:

- A trivial positive-class baseline
- Logistic regression with unigram bag-of-words features
- A Deep Averaging Network (DAN) using pretrained GloVe word embeddings

The program trains a selected model, reports accuracy, precision, recall, and F1 on the training and development sets, and can generate predictions for the blind test set.

## Requirements

- Python 3.9 or later
- NumPy
- PyTorch

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch
```

## Data and GloVe embeddings

The small train, development, and blind-test datasets are included in `data/`.

The pretrained GloVe files are not stored in this repository because of their size. Obtain the relativized embedding files from the course materials and place them in `data/` with these exact names:

```text
data/
├── train.txt
├── dev.txt
├── test-blind.txt
├── glove.6B.300d-relativized.txt
└── glove.6B.50d-relativized.txt     # only needed by test.py
```

The main program defaults to `data/glove.6B.300d-relativized.txt`. Although the linear and trivial models do not use the embeddings for prediction, the current entry point still loads the specified embedding file before selecting a model, so the 300-dimensional file must be present for all commands below.

## Usage

Run all commands from the repository root, where `sentiment_classifier.py` is located.

### Deep Averaging Network

The default configuration trains the DAN for 10 epochs with a learning rate of `0.001` and a hidden size of `100`:

```bash
python sentiment_classifier.py --model DAN --lr 0.001 --hidden_size 100 --num_epochs 10 --no_run_on_test
```

For a quicker smoke test, use one epoch:

```bash
python sentiment_classifier.py --model DAN --num_epochs 1 --no_run_on_test
```

### Logistic regression

The implemented linear-model configuration uses unigram features:

```bash
python sentiment_classifier.py --model LR --feats UNIGRAM --lr 0.1 --num_epochs 20 --no_run_on_test
```

`BIGRAM` and `BETTER` are scaffold options and are not implemented in the current code.

### Trivial baseline

```bash
python sentiment_classifier.py --model TRIVIAL --no_run_on_test
```

## Generating blind-test predictions

The `--no_run_on_test` flag skips writing blind-test predictions. Omit it when you want an output file:

```bash
python sentiment_classifier.py --model DAN
```

By default, predictions are written to `test-blind.output.txt`. To choose another path, use:

```bash
python sentiment_classifier.py --model DAN --test_output_path predictions.txt
```

Each output line contains the predicted label (`0` for negative or `1` for positive), a tab, and the tokenized sentence.

## Common command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `DAN` | Model to run: `TRIVIAL`, `LR`, or `DAN` |
| `--feats` | `UNIGRAM` | Feature type for logistic regression |
| `--lr` | `0.001` | Learning rate |
| `--num_epochs` | `10` | Number of training epochs |
| `--hidden_size` | `100` | DAN hidden-layer size |
| `--train_path` | `data/train.txt` | Labeled training data |
| `--dev_path` | `data/dev.txt` | Labeled development data |
| `--blind_test_path` | `data/test-blind.txt` | Unlabeled blind-test data |
| `--word_vecs_path` | `data/glove.6B.300d-relativized.txt` | Relativized GloVe file |
| `--test_output_path` | `test-blind.output.txt` | Prediction output path |
| `--no_run_on_test` | off | Skip generation of blind-test predictions |

You can view the complete option list with:

```bash
python sentiment_classifier.py --help
```

## Repository structure

```text
.
├── sentiment_classifier.py   # Command-line entry point and evaluation
├── models.py                 # Classifiers, feature extraction, and training
├── sentiment_data.py         # Dataset and embedding loading utilities
├── utils.py                  # Shared data structures and utilities
├── ffnn_example.py           # Small PyTorch feed-forward network example
├── test.py                   # Local embedding lookup experiment
└── data/                     # Train, development, test, and local embeddings
```

## Notes

- Run the program from the repository root so the default relative data paths resolve correctly.
- The DAN uses frozen pretrained embeddings, averages the token embeddings for each sentence, and passes the result through two linear layers with ReLU and dropout.
- Generated predictions, Python caches, virtual environments, IDE settings, model checkpoints, and GloVe files are excluded through `.gitignore`.
