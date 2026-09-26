# Question-Answering-NNs

Keras neural networks for **multiple-choice question answering with a supporting context**.
Each example is a `(context, question, candidate answer)` triple, and the network learns a binary
classifier that predicts whether the candidate answer is correct. At evaluation time the candidates
of a question are scored and the one with the highest "correct" probability is picked.

All models share the same input pipeline: the text is lowercased and lemmatized with NLTK, tokenized,
and embedded with frozen pre-trained 300-dimensional GloVe vectors.

## Repository layout

```
trainModel.py          Preprocessing (lemmas, tokenizer, embedding matrix) and training entry point
checkModel.py          Evaluates saved weights as 4-way multiple-choice accuracy on the test split
models/                Model architectures (one class per file, each with a train() method)
utils/
  splitDatum.py        Splits datum.txt into train_datum.txt / test_datum.txt
  data_helpers.py      POS-aware WordNet lemmatization (digits are replaced by the token "number")
  keras_utils.py       Custom Keras attention layers (AttentionLayer, AttentionLayerV2), TensorFlow only
  test_keras_attention.py  Sandbox that tries the attention layers on IMDb sentiment classification
  semantic_focus.py    Unfinished experiment (word2vec-based question focus extraction)
```

## Models

The name in the first column is the value passed to `trainModelOnFolder` in `trainModel.py`.

| Name            | Class                | Architecture |
|-----------------|----------------------|--------------|
| `simpleLSTM`    | `SimpleLSTMModel`    | BiLSTM(60) encoders for context, question and answer; question+context go through a Dense layer, then are combined with the answer and passed through another Dense layer before the softmax. |
| `cosLSTM`       | `CosLSTM`            | BiLSTM encoders; builds a (question, context) and an (answer, context) representation and classifies from their cosine similarity. |
| `noContextLSTM` | `NoContextLSTMModel` | BiLSTM encoders for question and answer only (context ignored), baseline. |
| `simpleCNN`     | `SimpleCNNModel`     | 1D convolutions with max pooling over each input (kernel sizes 3/5 for the answer, 5/7 for the question, 9 for the context), concatenated, Dense(100), softmax. |
| `cosCNN`        | `CosCNN`             | Same convolutional encoders as `simpleCNN`, combined through a cosine similarity. |
| `noContextCNN`  | `NoContextCNNModel`  | Convolutional encoders for question and answer only, baseline. |
| `LSTMwithCNN`   | `LSTMwithCNN`        | BiLSTM over context and question followed by convolutions on the LSTM outputs; BiLSTM over the answer. |
| –               | `LayeredLSTM`        | Like `simpleLSTM` but with a two-layer BiLSTM over the context. Not wired into `trainModel.py`. |

Every model outputs a 2-way softmax (incorrect / correct), is trained with categorical cross-entropy
and Nadam, and keeps the checkpoint with the best validation accuracy.

Input lengths are fixed (sequences are padded / truncated): question 100 tokens, answer 20 tokens,
context 500 tokens. The vocabulary is capped at 40,000 words.

## Dataset format

A dataset is a tab-separated file with a header row and the following columns:

| Column     | Content |
|------------|---------|
| `question` | Question text (no `\n` or `\t`) |
| `answer`   | Candidate answer text (same restriction) |
| `context`  | Supporting text for the question (same restriction) |
| `value`    | `1` if the answer is correct for the question, `0` otherwise |

Each question is expected to appear with **4 candidate answers, exactly one of them correct**.

Place each dataset in its own folder in the project root:

```
./
  glove.6B.300d.txt        Pre-trained GloVe embeddings (download separately)
  dataset_name/
    data/
      datum.txt            The dataset
    structures/            Tokenizer, embedding matrix, model JSON and weights are written here
```

## Usage

The scripts use hard-coded dataset folders (`data_small/` by default); edit the paths in the
`__main__` blocks to point at your dataset.

1. **Split the data** – run `utils/splitDatum.py` from inside `utils/` (it reads `../data_small/data/datum.txt`).
   - The first 400 rows become the test set, regrouped as one correct answer plus three incorrect ones per question.
   - The remaining rows become the training set, balanced by pairing every incorrect example with a correct one.
2. **Preprocess** – call `preprocessData('dataset_name/')` in `trainModel.py` once per new dataset. It saves
   the lemmatized texts (`train_lemmas_*`, `test_lemmas_*`), the tokenizer and the GloVe embedding matrix.
   Lemmatization is slow, so expect it to take around 20–30 minutes.
3. **Train** – call `trainModelOnFolder('<model name>', 'dataset_name/')` in `trainModel.py`. 10% of the
   training data is held out for validation. The architecture is saved to `structures/<prefix>-model1.json`
   and the best weights to `structures/<prefix>1-final-<epoch>-<val_acc>.hdf5`.
4. **Evaluate** – in `checkModel.py`, call `checkModel('<model json>', 'dataset_name/', '<weights file>')`.
   It prints the number of correctly answered questions, the total number of questions and the accuracy,
   where a question counts as correct when the true answer gets the highest score among its 4 candidates.

## Requirements

Python 3.5. Install the pinned dependencies with:

```
pip install -r requirements.txt
```

- Keras is pinned below 2.2 because the cosine models use `keras.layers.merge(mode='cos')`, which Keras 2.2.0 removed.
- The custom attention layers only work with the TensorFlow backend.
- gensim is only needed by `utils/semantic_focus.py`.
- NLTK needs the `wordnet` and `averaged_perceptron_tagger` data packages:
  `python -m nltk.downloader wordnet averaged_perceptron_tagger`
- The GloVe `glove.6B.300d.txt` embeddings must be in the project root.
