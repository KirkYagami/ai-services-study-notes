```markdown
# TOKENIZATION ANALYSIS AND COMPARISON FOR CUSTOMER FEEDBACK

## Problem Statement
Organizations receive customer feedback from multiple channels such as emails, surveys, support tickets, and online reviews. This feedback is stored in *unstructured text format*, making direct analysis challenging. To extract meaningful insights, the text must first undergo *tokenization*, a foundational Natural Language Processing (NLP) step that splits text into smaller units called tokens.

Different tokenization techniques handle linguistic constructs such as *contractions, punctuation, URLs, email addresses, abbreviations, emoticons, and special characters* in different ways. Selecting the right tokenization approach is critical for ensuring accuracy and consistency in downstream NLP tasks such as sentiment analysis, topic modeling, and text classification.

This project implements a *comparative tokenization analysis system* that applies *three distinct tokenization approaches* to the same dataset of customer feedback, applies *Porter stemming*, and generates a *comprehensive CSV-based comparison report*. The output enables data scientists and NLP practitioners to evaluate the strengths and limitations of each tokenization method.

---

## File Structure

```text
Project/
│
├── data/
│   └── customer_feedback.csv
├── outputs/
│   └── (generated during execution)
├── main.py
├── tests.py
└── requirements.txt

```

---

## Purpose of Files

* **`data/customer_feedback.csv`**
Contains 100 customer feedback samples with the columns `customer_id` and `feedback_text`.
* **`outputs/`**
Directory where the generated tokenization analysis CSV is written during execution.
* **`main.py`**
Core module that implements the `analyze_tokenization` function to perform comparative tokenization and stemming analysis.
* **`requirements.txt`**
Lists all external Python dependencies such as NLTK, spaCy, and pandas with version specifications.

---

## Module Explanations

### Module: `main.py`

#### Purpose

Performs comparative tokenization analysis using **NLTK word tokenizer**, **NLTK TreeBank tokenizer**, and **spaCy tokenizer**, applies **Porter Stemming**, and generates a structured CSV output.

#### Initialization Logic (Executed at Module Load)

* Downloads NLTK `punkt` tokenizer data silently
* Downloads NLTK `wordnet` data silently
* Loads spaCy English model `en_core_web_sm`
* If the spaCy model is unavailable, prints installation instructions and exits gracefully

---

### Function: `analyze_tokenization`

#### Function Signature

```python
analyze_tokenization(input_file_path, output_file_path)

```

#### Input Parameters

* **`input_file_path` (String):**
* Path to the input CSV file
* Must contain `customer_id` and `feedback_text` columns


* **`output_file_path` (String):**
* Path where the analysis CSV will be written
* Must end with `.csv`



#### Return Type

`pandas.DataFrame`

#### Return Description

Returns a `DataFrame` with **100 rows and 11 columns**, representing tokenization and stemming metrics for each feedback record. The same data is also written to a CSV file.

---

## Function Logic and Workflow

### Step 1: Data Loading

* Opens input CSV file using `pandas.read_csv` function
* Extracts `feedback_text` column from CSV
* Converts column to Python list for iteration
* Total items in list: 100 customer feedback strings

### Step 2: Tokenizer Initialization

* Creates `TreebankWordTokenizer` instance from NLTK library
* Creates `PorterStemmer` instance from NLTK library
* These objects will be reused for all 100 feedback samples

### Step 3: Iterative Processing Loop

Iterates through each of 100 feedback texts with enumeration starting at 1. For each feedback text (iteration $i$, where $1 \le i \le 100$):

#### Tokenization Substep 3a: NLTK Word Tokenization

* Converts feedback text to lowercase
* Applies NLTK `word_tokenize` function
* Produces list of individual tokens (words and punctuation)
* *Example:* `"Don't worry"` becomes `["Do", "n't", "worry"]`

#### Tokenization Substep 3b: TreeBank Tokenization

* Converts feedback text to lowercase
* Applies `TreebankWordTokenizer.tokenize` method
* Produces list preserving contractions and punctuation
* *Example:* `"Don't worry"` becomes `["Do", "n't", "worry"]`

#### Tokenization Substep 3c: spaCy Tokenization

* Converts feedback text to lowercase
* Processes through spaCy pipeline (pre-loaded model)
* Extracts `.text` attribute from each `Token` object
* Produces list of string tokens
* *Example:* `"Don't worry"` becomes `["do", "n't", "worry"]` or `["don't", "worry"]` depending on model

#### Stemming Substep 3d: Apply PorterStemmer to All Three Token Lists

* Applies `stemmer.stem()` to each token from NLTK tokens
* Applies `stemmer.stem()` to each token from TreeBank tokens
* Applies `stemmer.stem()` to each token from spaCy tokens
* Produces three stem lists (same length as token lists)
* *Example:* `"running"` becomes `"run"`, `"happiness"` becomes `"happi"`

#### Collection Substep 3e: Aggregate Results

* Records `feedback_id` as current iteration number
* Records `original_text` as first 50 characters of feedback
* Records token counts as `len(tokens_list)` for each method
* Records sample tokens as first 5 tokens joined with pipe (`|`) separator
* Records sample stems as first 5 stems joined with pipe (`|`) separator
* Creates dictionary with 11 key-value pairs

### Step 4: DataFrame Creation

* Converts list of 100 dictionaries into `pandas.DataFrame`
* Automatically creates 11 columns from dictionary keys
* Creates 100 rows (one per feedback sample)

### Step 5: CSV Export

* Writes DataFrame to output file path provided as parameter
* Uses `UTF-8-sig` encoding (includes BOM for Excel compatibility)
* Uses `quoting=1` setting (`QUOTE_ALL`) to handle pipe characters in token samples
* Excludes index column (no row numbers)
* Ensures proper handling of special characters and delimiters

**Return:** Returns the created `DataFrame` object for further use in main block.

---

## Entry Point

**Execution Type:** Module-level `if __name__ == "__main__":` block

**Hardcoded File Paths:**

* Input: `"data/customer_feedback.csv"`
* Output: `"outputs/analysis_results.csv"`

**Execution Steps:**

1. Calls `analyze_tokenization` with hardcoded paths
2. Receives returned `DataFrame`
3. Prints completion message with output file location
4. Prints number of processed samples using `len(results)`

**Console Output Example:**

```text
Analysis complete. Results saved to: outputs/analysis_results.csv
Processed 100 feedback samples

```

---

## Input and Output Specifications

### Input Format: CSV File

* **File Name:** `customer_feedback.csv`
* **Location:** `data/` directory
* **Delimiter:** Comma
* **Encoding:** UTF-8 (default)
* **Header Row:** Present (row 1)
* **Data Rows:** 100 rows (rows 2–101)
* **Columns:**
1. `customer_id` (Integer, range 1–100)
2. `feedback_text` (String, variable length)



### Output Format: CSV File

* **File Name:** `analysis_results.csv`
* **Location:** `outputs/` directory
* **Delimiter:** Comma
* **Encoding:** UTF-8-sig (includes BOM)
* **Quoting:** `QUOTE_ALL` (all fields quoted)
* **Header Row:** Present
* **Data Rows:** 100 rows (one per input feedback)

#### Expected Columns (11 total):

1. `feedback_id` (Integer, 1–100)
2. `original_text` (String, first 50 characters)
3. `nltk_tokens_count` (Integer, count of NLTK tokens)
4. `treebank_tokens_count` (Integer, count of TreeBank tokens)
5. `spacy_tokens_count` (Integer, count of spaCy tokens)
6. `nltk_sample_tokens` (String, first 5 NLTK tokens pipe-separated)
7. `treebank_sample_tokens` (String, first 5 TreeBank tokens pipe-separated)
8. `spacy_sample_tokens` (String, first 5 spaCy tokens pipe-separated)
9. `nltk_sample_stems` (String, first 5 NLTK stems pipe-separated)
10. `treebank_sample_stems` (String, first 5 TreeBank stems pipe-separated)
11. `spacy_sample_stems` (String, first 5 spaCy stems pipe-separated)

---

## Expected Output Structure

### Sample Output CSV (`analysis_results.csv`)

| feedback_id | original_text | nltk_tokens_count | treebank_tokens_count | spacy_tokens_count | nltk_sample_tokens | treebank_sample_tokens | spacy_sample_tokens | nltk_sample_stems | treebank_sample_stems | spacy_sample_stems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| "1" | "The product quality is excellent, but delivery was " | 12 | 12 | 11 | "the|product|quality|is|excellent" | "the|product|quality|is|excellent" | "the|product|quality|is|excellent" | "the|product|qualiti|is|excel" | "the|product|qualiti|is|excel" | "the|product|qualiti|is|excel" |
| "2" | "Customer support didn't answer my questions promptly" | 10 | 10 | 9 | "customer|support|did|n't|answer" | "customer|support|did|n't|answer" | "customer|support|didn't|answer|my" | "custom|support|did|n't|answer" | "custom|support|did|n't|answer" | "custom|support|didn't|answer|my" |

---

## Implementation Instructions

### Environment Setup

1. **Navigate to project directory:**
```bash
cd Project

```


2. **Install dependencies from `requirements.txt`:**
```bash
pip install -r requirements.txt

```


3. **Install spaCy model separately (if needed):**
```bash
python3 -m spacy download en_core_web_sm

```


4. **Run Solution:**
```bash
python3 main.py

```


5. **Run Test Cases:**
```bash
python3 -m pytest tests.py -vv

```



```

```