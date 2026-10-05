"""Train an English-to-German LSTM encoder-decoder on Multi30k."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parent
MAX_LENGTH = 20
START = "<start>"
END = "<end>"


def load_parallel_data(directory, split):
    """Read aligned sentences without silently dropping unmatched lines."""
    source = (Path(directory) / f"{split}.en").read_text(encoding="utf-8").splitlines()
    target = (Path(directory) / f"{split}.de").read_text(encoding="utf-8").splitlines()
    if len(source) != len(target) or not source:
        raise ValueError(f"Invalid parallel corpus for {split}: {len(source)} / {len(target)} lines")
    if any(not sentence.strip() for sentence in source + target):
        raise ValueError(f"Empty sentence in {split}")
    return source, target


def build_tokenizers(english, german):
    """Fit separate vocabularies on training text, preserving punctuation."""
    source = Tokenizer(filters="", oov_token="<unk>")
    target = Tokenizer(filters="", oov_token="<unk>")
    source.fit_on_texts(english)
    target.fit_on_texts([f"{START} {sentence} {END}" for sentence in german])
    return source, target


def prepare_data(english, german, source_tokenizer, target_tokenizer):
    source = pad_sequences(source_tokenizer.texts_to_sequences(english),
                           maxlen=MAX_LENGTH, padding="post", truncating="post")
    # Reserve space for both boundary tokens, even when a caption is truncated.
    target_sequences = [
        [target_tokenizer.word_index[START]] + sequence[:MAX_LENGTH - 2]
        + [target_tokenizer.word_index[END]]
        for sequence in target_tokenizer.texts_to_sequences(german)
    ]
    target = pad_sequences(target_sequences, maxlen=MAX_LENGTH, padding="post")
    return source, target[:, :-1], target[:, 1:]


def build_models(source_vocab, target_vocab):
    """Share trained layers between teacher-forced and single-step graphs."""
    source_input = layers.Input(shape=(None,), dtype="int32", name="english")
    target_input = layers.Input(shape=(None,), dtype="int32", name="german")
    source_embedding = layers.Embedding(source_vocab, 256, mask_zero=True)
    encoder_lstm = layers.LSTM(512, return_state=True, name="encoder_lstm")
    _, hidden, cell = encoder_lstm(source_embedding(source_input))

    target_embedding = layers.Embedding(target_vocab, 256, mask_zero=True)
    decoder_lstm = layers.LSTM(512, return_sequences=True, return_state=True, name="decoder_lstm")
    output_layer = layers.Dense(target_vocab, activation="softmax", name="token_probabilities")
    sequence, _, _ = decoder_lstm(target_embedding(target_input), initial_state=[hidden, cell])
    model = Model([source_input, target_input], output_layer(sequence))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    encoder = Model(source_input, [hidden, cell], name="encoder")
    token = layers.Input(shape=(1,), dtype="int32", name="previous_token")
    previous_hidden = layers.Input(shape=(512,), name="previous_hidden")
    previous_cell = layers.Input(shape=(512,), name="previous_cell")
    output, next_hidden, next_cell = decoder_lstm(
        target_embedding(token), initial_state=[previous_hidden, previous_cell]
    )
    decoder = Model([token, previous_hidden, previous_cell],
                    [output_layer(output), next_hidden, next_cell], name="decoder")
    return model, encoder, decoder


def translate(sentence, source_tokenizer, target_tokenizer, encoder, decoder):
    """Greedily decode until the end token or the output length limit."""
    source = pad_sequences(source_tokenizer.texts_to_sequences([sentence]),
                           maxlen=MAX_LENGTH, padding="post", truncating="post")
    hidden, cell = encoder(source, training=False)
    token = target_tokenizer.word_index[START]
    words = []
    for _ in range(MAX_LENGTH):
        probabilities, hidden, cell = decoder(
            [tf.constant([[token]]), hidden, cell], training=False
        )
        token = int(tf.argmax(probabilities[0, 0]).numpy())
        if token in (0, target_tokenizer.word_index[END]):
            break
        word = target_tokenizer.index_word.get(token, "<unk>")
        if word not in {START, END, "<unk>"}:
            words.append(word)
    return " ".join(words)


def plot_history(history, directory):
    for metric in ("loss", "accuracy"):
        figure, axis = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], "o-", label="Training")
        axis.plot(epochs, history[f"val_{metric}"], "s-", label="Validation")
        axis.set(title=f"English–German Translation: {metric.capitalize()}",
                 xlabel="Epoch", ylabel=f"Non-padding token {metric}")
        axis.set_xticks(list(epochs))
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"translation_{metric}.png", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Optional training sample limit for smoke checks")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Epochs, batch size and sample limit must be positive")
    tf.keras.utils.set_random_seed(42)
    train_en, train_de = load_parallel_data(args.data_dir, "train")
    val_en, val_de = load_parallel_data(args.data_dir, "val")
    test_en, test_de = load_parallel_data(args.data_dir, "test")
    if args.limit:
        train_en, train_de = train_en[:args.limit], train_de[:args.limit]
        val_en, val_de = val_en[:args.limit], val_de[:args.limit]
    source_tokenizer, target_tokenizer = build_tokenizers(train_en, train_de)
    train = prepare_data(train_en, train_de, source_tokenizer, target_tokenizer)
    validation = prepare_data(val_en, val_de, source_tokenizer, target_tokenizer)
    test = prepare_data(test_en, test_de, source_tokenizer, target_tokenizer)
    print(f"Pairs: train={len(train_en)}, validation={len(val_en)}, test={len(test_en)}", flush=True)
    model, encoder, decoder = build_models(len(source_tokenizer.word_index) + 1,
                                           len(target_tokenizer.word_index) + 1)
    model.summary()
    history = model.fit(
        [train[0], train[1]], train[2], sample_weight=(train[2] != 0).astype("float32"),
        validation_data=([validation[0], validation[1]], validation[2],
                         (validation[2] != 0).astype("float32")),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.output_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for name, tokenizer in (("english", source_tokenizer), ("german", target_tokenizer)):
        (artifacts / f"{name}_tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    encoder.save(artifacts / "encoder.keras")
    decoder.save(artifacts / "decoder.keras")
    (args.output_dir / "training_history.json").write_text(json.dumps(history.history, indent=2) + "\n")
    plot_history(history.history, args.output_dir)
    scores = model.evaluate([test[0], test[1]], test[2],
                            sample_weight=(test[2] != 0).astype("float32"),
                            batch_size=args.batch_size, return_dict=True, verbose=0)
    (args.output_dir / "evaluation.json").write_text(json.dumps(scores, indent=2) + "\n")
    examples = []
    for english, german in zip(test_en[:5], test_de[:5]):
        predicted = translate(english, source_tokenizer, target_tokenizer, encoder, decoder)
        examples.append({"english": english, "reference": german, "prediction": predicted})
        print(f"\nEnglish: {english}\nReference: {german}\nPrediction: {predicted}", flush=True)
    (args.output_dir / "translations.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    print(f"Test token metrics: {scores}")
    print(f"Saved plots, metrics, tokenizers and inference models to {args.output_dir}")


if __name__ == "__main__":
    main()
