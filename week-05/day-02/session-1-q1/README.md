# English-to-German translation with an LSTM encoder-decoder

Run from the repository root:

```bash
.venv/bin/python -m pip install -r week-05/day-02/session-1-q1/requirements.txt
.venv/bin/python week-05/day-02/session-1-q1/main.py
```

The default run uses all 29,000 training pairs, 1,014 validation pairs and 1,000
test pairs. It trains for 10 epochs with batch size 64, 256-dimensional embeddings,
and 512-unit encoder/decoder LSTMs. This is a substantial CPU training job.

The English and German tokenizers are fitted only on the training split. They
preserve punctuation, lowercase text, and map unseen words to `<unk>`. English
sequences are post-padded/truncated to 20 tokens. German sequences reserve two
of those positions for `<start>` and `<end>` before shifting to form the 19-step
teacher-forcing input and target. Padding is excluded from loss and accuracy.
Accuracy measures next-token predictions with teacher forcing, not the proportion
of completely correct translations.

Data belongs in `data/{train,val,test}.{en,de}`. The dataset, source ZIP and trained
artifacts are ignored by Git. The ZIP has been extracted locally.

Outputs:

- `translation_loss.png` and `translation_accuracy.png`: learning curves.
- `training_history.json` and `evaluation.json`: actual epoch and test metrics.
- `translations.json`: five held-out captions, references and greedy predictions.
- `artifacts/english_tokenizer.json` and `artifacts/german_tokenizer.json`.
- `artifacts/encoder.keras` and `artifacts/decoder.keras`: inference models.
- `ANALYSIS.txt`: interpretation of the completed run.

A small verification run can use `--limit 64 --epochs 1 --output-dir /tmp/translation-smoke`.
The sample limit is for debugging and is not the assignment's full training run.

To reload inference assets, import `tokenizer_from_json` from
`tensorflow.keras.preprocessing.text`, load each tokenizer JSON string with it,
and load both models with `tf.keras.models.load_model(path, compile=False)`.
Pass them to `translate(sentence, english_tokenizer, german_tokenizer, encoder, decoder)`
in `main.py`. Greedy decoding stops at the end token or after 20 steps.
