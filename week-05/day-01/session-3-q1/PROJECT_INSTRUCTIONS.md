# SENTIMENT ANALYSIS OF MOVIE REVIEWS USING STACKED LSTM NETWORKS

## PROBLEM STATEMENT
A digital media analytics company requires an automated sentiment classification system to analyze movie reviews from online platforms[cite: 1]. The system must process variable-length text reviews, learn contextual patterns using deep learning, and classify sentiments as positive or negative with comprehensive training performance visualization for model evaluation and optimization[cite: 1].

---

## OBJECTIVE
The primary objective is to implement a binary sentiment classification system using stacked Long Short-Term Memory networks trained on movie review text data[cite: 1]. Specifically, the system must:

1. Load and organize movie review text files from positive and negative sentiment directories[cite: 1].
2. Implement text tokenization to convert reviews into numerical sequences suitable for neural network processing[cite: 1].
3. Apply sequence padding to standardize variable-length reviews to uniform input dimensions[cite: 1].
4. Build a stacked LSTM architecture with embedding layer for word representation learning[cite: 1].
5. Train the binary classifier with dropout regularization to prevent overfitting[cite: 1].
6. Monitor training and validation performance across multiple epochs to assess model convergence[cite: 1].
7. Generate comprehensive dual-panel visualization showing accuracy and loss curves[cite: 1].
8. Compare training versus validation metrics to identify potential overfitting or underfitting patterns[cite: 1].

---

## DATASET DESCRIPTION
The project utilizes the IMDB Movie Reviews dataset, a widely recognized benchmark corpus for binary sentiment classification tasks containing authentic user-generated movie reviews[cite: 1].

* **Dataset Name:** IMDB Movie Reviews Dataset[cite: 1]
* **File Format:** Individual text files (`.txt`) organized in sentiment-labeled directories[cite: 1]
* **Total Available Reviews:** 50,000 labeled reviews[cite: 1]
* **Used for Training:** 10,000 reviews (5,000 positive + 5,000 negative)[cite: 1]
* **Used for Testing:** 2,000 reviews (1,000 positive + 1,000 negative)[cite: 1]
* **Encoding:** UTF-8[cite: 1]

### Data Organization
```text
data/
├── train/
│   ├── pos/   # Positive sentiment training reviews
│   ├── neg/   # Negative sentiment training reviews
│   └── unsup/ # Unsupervised reviews (not used in this project)
└── test/
    ├── pos/   # Positive sentiment test reviews
    └── neg/   # Negative sentiment test reviews
```[cite: 1]

### Directory Contents
* **Training Positive Reviews:** 12,500 files available (uses 5,000)[cite: 1]
* **Training Negative Reviews:** 12,500 files available (uses 5,000)[cite: 1]
* **Training Unsupervised:** 50,000 files (unused in this project)[cite: 1]
* **Test Positive Reviews:** 12,500 files available (uses 1,000)[cite: 1]
* **Test Negative Reviews:** 12,500 files available (uses 1,000)[cite: 1]

### Review Characteristics
* **File Naming Convention:** `{review_id}_{rating}.txt`[cite: 1]
* **Content Type:** Natural English language movie reviews[cite: 1]
* **Review Length:** Variable (ranging from single sentences to multiple paragraphs)[cite: 1]
* **Language Style:** Informal user-generated content with opinions, emotions, and subjective assessments[cite: 1]
* **Sentiment Labels:** Binary classification (positive=1, negative=0)[cite: 1]

#### Sample Review Structure (Positive)
> "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that..."[cite: 1]

#### Sample Review Structure (Negative)
> "Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting..."[cite: 1]

### Text Preprocessing Requirements
* **Vocabulary Size:** 10,000 most frequent words[cite: 1]
* **Sequence Length:** 200 tokens (pad shorter reviews, truncate longer reviews)[cite: 1]
* **Tokenization:** Word-level tokenization with integer encoding[cite: 1]
* **Padding Strategy:** Post-padding with zeros to achieve uniform length[cite: 1]
* **Out-of-Vocabulary Handling:** Words beyond 10,000 vocabulary limit are ignored[cite: 1]

---

## TASKS

### Task 1: Environment Setup and Random Seed Configuration
Configure the Python environment with essential libraries including TensorFlow, Keras, NumPy, and Matplotlib[cite: 1]. Set random seeds for both NumPy and TensorFlow to ensure reproducible results across multiple training runs[cite: 1]. This reproducibility is critical for debugging, model comparison, and consistent performance evaluation[cite: 1].

### Task 2: Data Loading Function Development
Create functions to systematically load movie review text files from the organized directory structure[cite: 1]. Develop separate loading functions for training and test datasets that traverse the positive and negative sentiment directories, read individual text files, extract review content, assign appropriate binary labels, and return lists of text strings paired with their sentiment labels[cite: 1]. Implement configurable sample limits to control dataset size for computational efficiency[cite: 1].

### Task 3: Training and Test Data Loading
Execute the data loading functions to read IMDB movie reviews from the file system[cite: 1]. Load 10,000 training samples consisting of equal numbers of positive and negative reviews to maintain class balance[cite: 1]. Load 2,000 test samples with balanced positive and negative representation for unbiased validation[cite: 1]. Display dataset statistics including total training and test sample counts to confirm successful data loading[cite: 1].

### Task 4: Text Tokenization and Vocabulary Building
Implement text tokenization to convert raw review strings into numerical sequences suitable for neural network processing[cite: 1]. Create a tokenizer that analyzes the training text corpus and builds a vocabulary of the 10,000 most frequently occurring words[cite: 1]. Fit the tokenizer exclusively on training data to prevent data leakage[cite: 1]. Convert both training and test review texts into sequences of integer indices where each integer represents a specific word from the learned vocabulary[cite: 1].

### Task 5: Sequence Padding and Label Preparation
Apply sequence padding to standardize all reviews to a uniform length of 200 tokens[cite: 1]. Pad shorter reviews with zeros to reach the target length and truncate longer reviews to maintain consistency[cite: 1]. This standardization enables batch processing in the neural network[cite: 1]. Convert sentiment label lists to NumPy arrays for efficient training[cite: 1]. The result is fixed-size numerical matrices ready for LSTM input[cite: 1].

### Task 6: Stacked LSTM Model Architecture Design
Construct a sequential deep learning model implementing a stacked LSTM architecture for hierarchical sequence learning[cite: 1]. The architecture should begin with an embedding layer that converts integer word indices into dense 128-dimensional vector representations, enabling the model to learn semantic word relationships[cite: 1]. Follow with a first LSTM layer containing 128 units configured to return full sequences rather than just final outputs, allowing information flow to subsequent layers[cite: 1]. Add dropout regularization to prevent overfitting[cite: 1]. Include a second LSTM layer with 64 units processing the sequence to extract high-level features, followed by additional dropout[cite: 1]. Incorporate a fully connected dense layer with 64 units and ReLU activation for non-linear feature transformation, more dropout, and conclude with a single-unit output layer using sigmoid activation to produce probability scores for binary classification[cite: 1].

### Task 7: Model Compilation Configuration
Configure the model training process by specifying the optimization algorithm, loss function, and evaluation metrics[cite: 1]. Use the Adam optimizer for adaptive learning rate adjustment during training[cite: 1]. Apply binary cross-entropy loss function appropriate for two-class classification problems[cite: 1]. Include accuracy as the primary evaluation metric to monitor the percentage of correctly classified reviews during training and validation[cite: 1].

### Task 8: Model Training with Validation Monitoring
Execute the training process for the specified number of epochs using the prepared training data[cite: 1]. Configure batch processing with an appropriate batch size to balance memory efficiency and gradient stability[cite: 1]. Provide the test dataset as validation data to monitor generalization performance during training[cite: 1]. Enable verbose output to display epoch-by-epoch progress showing training loss, training accuracy, validation loss, and validation accuracy[cite: 1]. Capture the complete training history object containing all metrics across epochs for subsequent analysis and visualization[cite: 1].

### Task 9: Training History Visualization Creation
Generate a comprehensive dual-panel visualization to illustrate model learning progression and performance characteristics[cite: 1]. Create a side-by-side subplot layout with the left panel displaying accuracy curves and the right panel showing loss curves[cite: 1]. For each panel, plot both training and validation metrics across all epochs using distinct colors and markers for clear differentiation[cite: 1]. Apply appropriate axis labels, titles, legends, and grid lines for professional presentation[cite: 1]. Include a summary text box displaying final training and validation accuracy values to provide quantitative performance assessment[cite: 1].

### Task 10: Visualization Formatting and Export
Apply professional formatting to the complete figure including an overall title identifying the task as LSTM sentiment analysis training history[cite: 1]. Configure adequate figure dimensions for clarity and readability of both panels[cite: 1]. Use tight layout to optimize spacing and prevent element overlap[cite: 1]. Adjust subplot positioning to accommodate the final metrics text box at the bottom of the figure[cite: 1]. Save the visualization as a high-resolution PNG image file at 150 DPI with tight bounding box to eliminate excess whitespace, creating a publication-quality output suitable for technical reports and presentations[cite: 1].

### Task 11: Results Reporting and Model Evaluation
Display completion messages indicating successful visualization export with output filename[cite: 1]. Report the final validation accuracy as the primary metric representing model performance on unseen test data[cite: 1]. This validation accuracy provides an objective assessment of the model's ability to generalize sentiment classification to new reviews beyond the training set, indicating real-world deployment viability[cite: 1].

---

## TECHNICAL SPECIFICATIONS

### Dataset Configuration
* **Training samples:** 10,000 (5,000 positive + 5,000 negative)[cite: 1]
* **Test samples:** 2,000 (1,000 positive + 1,000 negative)[cite: 1]
* **Vocabulary size:** 10,000 most frequent words[cite: 1]
* **Sequence length:** 200 tokens (padded/truncated)[cite: 1]
* **Embedding dimension:** 128[cite: 1]

### Model Architecture
* **Embedding Layer:** 10,000 vocab $\rightarrow$ 128 dimensions[cite: 1]
* **LSTM Layer 1:** 128 units, `return_sequences=True`[cite: 1]
* **Dropout:** 0.2[cite: 1]
* **LSTM Layer 2:** 64 units[cite: 1]
* **Dropout:** 0.2[cite: 1]
* **Dense Layer:** 64 units, ReLU activation[cite: 1]
* **Dropout:** 0.2[cite: 1]
* **Output Layer:** 1 unit, sigmoid activation[cite: 1]

### Training Configuration
* **Batch size:** 128[cite: 1]
* **Epochs:** 10[cite: 1]
* **Optimizer:** Adam[cite: 1]
* **Loss:** `binary_crossentropy`[cite: 1]
* **Metrics:** `accuracy`[cite: 1]
* **Validation:** Test set during training[cite: 1]

### Visualization Specifications
* **Layout:** $1 \times 2$ subplot (side-by-side)[cite: 1]
* **Figure size:** $14 \times 5$ inches[cite: 1]
* **Panels:** Accuracy (left), Loss (right)[cite: 1]
* **Resolution:** 150 DPI[cite: 1]
* **Output file:** `sentiment_training_history.png`[cite: 1]

---

## DELIVERABLES

### 1) Training History Visualization
High-resolution PNG image file (`sentiment_training_history.png`, approximately 107 KB at 150 DPI) displaying dual-panel horizontal layout[cite: 1]. Left panel shows Model Accuracy with training accuracy curve rising from 71% to 97% and validation accuracy curve rising to 83% then declining to 81%, demonstrating overfitting pattern[cite: 1]. Right panel shows Model Loss with training loss steadily decreasing from 0.55 to 0.08 while validation loss increases after initial decrease, further confirming overfitting[cite: 1]. Bottom text box displays final metrics showing 97.27% training accuracy versus 80.70% validation accuracy[cite: 1]. Professional formatting includes overall title, axis labels, legends distinguishing training and validation curves, grid lines, and appropriate color scheme for clear data visualization[cite: 1].

### 2) ANALYSIS REPORT
* **Filename:** `ANALYSIS.txt` or `ANALYSIS.pdf`[cite: 1]
* **Description:** Model performance analysis[cite: 1]
* **Length:** 300–400 words[cite: 1]