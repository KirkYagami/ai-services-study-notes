# IMDB sentiment analysis with stacked LSTMs

The program loads balanced positive and negative movie reviews, learns a vocabulary
from training reviews only, and trains the specified stacked LSTM classifier.
Zero padding is masked so it does not become review content.

From the repository root:

```bash
.venv/bin/python -m pip install -r week-05/day-01/session-3-q1/requirements.txt
.venv/bin/python week-05/day-01/session-3-q1/main.py
```

The default run uses 5,000 training reviews and 1,000 validation reviews per class,
10 epochs, and a batch size of 128. The vocabulary limit is 10,000 and sequences
are post-padded or post-truncated to 200 tokens. Files are sorted before sampling,
and NumPy and TensorFlow use seed 42. Exact results may vary with hardware and
library versions.

The local dataset must contain `data/train/{pos,neg}` and `data/test/{pos,neg}`.
The data directory is ignored by Git; the extracted ZIP has been removed.
Unsupervised reviews are not used.

Outputs are saved beside `main.py`:

- `sentiment_training_history.png`: training and validation accuracy/loss curves.
- `training_history.json`: measured metrics from each epoch.
- `ANALYSIS.txt`: discussion of the completed default training run.

For a quick smoke run, use `--epochs 1 --train-limit 32 --test-limit 16`.
This overwrites the chart and JSON with the smaller run's results.

The assignment uses the test split for validation during training. Consequently,
it is a monitored validation set, not an independent final evaluation set.
