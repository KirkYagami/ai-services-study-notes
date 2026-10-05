"""Train a stacked LSTM on IMDB reviews and plot its learning curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

PROJECT_DIR = Path(__file__).resolve().parent
VOCAB_SIZE = 10_000
SEQUENCE_LENGTH = 200


def load_reviews(directory, limit):
    """Read up to limit reviews per sentiment in a repeatable file order."""
    reviews, labels = [], []
    for sentiment, label in (("pos", 1), ("neg", 0)):
        folder = Path(directory) / sentiment
        files = sorted(folder.glob("*.txt"))[:limit]
        if not files:
            raise ValueError(f"No reviews found in {folder}")
        for file in files:
            reviews.append(file.read_text(encoding="utf-8"))
            labels.append(label)
    return reviews, labels


def load_training_data(data_dir, limit=5_000):
    return load_reviews(Path(data_dir) / "train", limit)


def load_test_data(data_dir, limit=1_000):
    return load_reviews(Path(data_dir) / "test", limit)


def prepare_sequences(train_reviews, test_reviews):
    """Learn the vocabulary on training reviews only and post-pad sequences."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_reviews)
    train = pad_sequences(tokenizer.texts_to_sequences(train_reviews),
                          maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")
    test = pad_sequences(tokenizer.texts_to_sequences(test_reviews),
                         maxlen=SEQUENCE_LENGTH, padding="post", truncating="post")
    return train, test


def build_model():
    """Build and compile the specified two-layer LSTM classifier."""
    model = Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH,), dtype="int32"),
        layers.Embedding(VOCAB_SIZE, 128, mask_zero=True),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history, output_path):
    """Save accuracy and loss panels with the final measured accuracies."""
    metrics = history.history
    epochs = range(1, len(metrics["accuracy"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric, title in zip(axes, ("accuracy", "loss"),
                                   ("Model Accuracy", "Model Loss")):
        axis.plot(epochs, metrics[metric], "o-", label="Training", color="tab:blue")
        axis.plot(epochs, metrics[f"val_{metric}"], "s-", label="Validation", color="tab:orange")
        axis.set(title=title, xlabel="Epoch", ylabel=metric.capitalize())
        axis.set_xticks(list(epochs))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("LSTM Sentiment Analysis Training History")
    figure.text(0.5, 0.025,
                f"Final training accuracy: {metrics['accuracy'][-1]:.2%}    |    "
                f"Final validation accuracy: {metrics['val_accuracy'][-1]:.2%}",
                ha="center", bbox={"facecolor": "#eef2f6", "edgecolor": "#ccd3da", "pad": 7})
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=5_000, help="Reviews per sentiment")
    parser.add_argument("--test-limit", type=int, default=1_000, help="Reviews per sentiment")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_limit, args.test_limit) < 1:
        parser.error("Epochs, batch size, and sample limits must be positive.")

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    train_reviews, train_labels = load_training_data(args.data_dir, args.train_limit)
    test_reviews, test_labels = load_test_data(args.data_dir, args.test_limit)
    print(f"Training reviews: {len(train_reviews)} | Test reviews: {len(test_reviews)}", flush=True)
    train, test = prepare_sequences(train_reviews, test_reviews)
    model = build_model()
    model.summary()
    history = model.fit(
        train, np.asarray(train_labels), epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(test, np.asarray(test_labels)), shuffle=True, verbose=2,
    )
    output = PROJECT_DIR / "sentiment_training_history.png"
    plot_history(history, output)
    # Keep measured values available for the written analysis and reproducibility.
    (PROJECT_DIR / "training_history.json").write_text(
        json.dumps(history.history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved visualization: {output.name}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
