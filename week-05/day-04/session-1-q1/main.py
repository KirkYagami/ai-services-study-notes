import os
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import PorterStemmer

# ----------------------------------------------------
# Initialization Logic (Executed at Module Load)
# ----------------------------------------------------
# Download required NLTK datasets silently
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy English model en_core_web_sm
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    print("Error: The spaCy model 'en_core_web_sm' is unavailable.")
    print("Please install it using: python3 -m spacy download en_core_web_sm")
    sys.exit(1)


# ----------------------------------------------------
# Main Function Implementation
# ----------------------------------------------------
def analyze_tokenization(input_file_path, output_file_path):
    """
    Performs comparative tokenization and stemming analysis using:
    - NLTK word tokenizer
    - NLTK TreeBank tokenizer
    - spaCy tokenizer

    Applies Porter stemming to all tokens, records metrics, and saves to CSV.
    """
    # Step 1: Data Loading
    df_input = pd.read_csv(input_file_path)
    feedback_texts = df_input['feedback_text'].tolist()

    # Step 2: Tokenizer Initialization
    treebank_tokenizer = TreebankWordTokenizer()
    stemmer = PorterStemmer()

    results = []

    # Step 3: Iterative Processing Loop
    for i, text in enumerate(feedback_texts, start=1):
        text_str = str(text) if pd.notna(text) else ""
        text_lower = text_str.lower()

        # Tokenization Substep 3a: NLTK Word Tokenization
        nltk_tokens = word_tokenize(text_lower)

        # Tokenization Substep 3b: TreeBank Tokenization
        treebank_tokens = treebank_tokenizer.tokenize(text_lower)

        # Tokenization Substep 3c: spaCy Tokenization
        doc = nlp(text_lower)
        spacy_tokens = [token.text for token in doc]

        # Stemming Substep 3d: Apply PorterStemmer to all token lists
        nltk_stems = [stemmer.stem(token) for token in nltk_tokens]
        treebank_stems = [stemmer.stem(token) for token in treebank_tokens]
        spacy_stems = [stemmer.stem(token) for token in spacy_tokens]

        # Collection Substep 3e: Aggregate Results
        row_dict = {
            'feedback_id': i,
            'original_text': text_str[:50],
            'nltk_tokens_count': len(nltk_tokens),
            'treebank_tokens_count': len(treebank_tokens),
            'spacy_tokens_count': len(spacy_tokens),
            'nltk_sample_tokens': "|".join(nltk_tokens[:5]),
            'treebank_sample_tokens': "|".join(treebank_tokens[:5]),
            'spacy_sample_tokens': "|".join(spacy_tokens[:5]),
            'nltk_sample_stems': "|".join(nltk_stems[:5]),
            'treebank_sample_stems': "|".join(treebank_stems[:5]),
            'spacy_sample_stems': "|".join(spacy_stems[:5])
        }
        results.append(row_dict)

    # Step 4: DataFrame Creation
    output_df = pd.DataFrame(results)

    # Step 5: CSV Export
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_df.to_csv(
        output_file_path,
        encoding='utf-8-sig',
        quoting=1,  # csv.QUOTE_ALL
        index=False
    )

    return output_df


# ----------------------------------------------------
# Entry Point
# ----------------------------------------------------
if __name__ == "__main__":
    input_path = "data/customer_feedback.csv"
    output_path = "outputs/analysis_results.csv"

    results_df = analyze_tokenization(input_path, output_path)

    print(f"Analysis complete. Results saved to: {output_path}")
    print(f"Processed {len(results_df)} feedback samples")