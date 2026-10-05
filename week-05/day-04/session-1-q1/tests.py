import pandas as pd
import os
import tempfile
import csv
from main import analyze_tokenization


def test_output_file_creation():
    """Test 1: Verify CSV output file is created"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    analyze_tokenization(input_path, output_path)
    assert os.path.exists(output_path), f"Output file was not created at {output_path}"


def test_row_count_match():
    """Test 2: Verify output rows equal input rows"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    input_df = pd.read_csv(input_path)
    assert len(result) == len(input_df), f"Row count mismatch: expected {len(input_df)} but got {len(result)}"


def test_required_columns():
    """Test 3: Verify all necessary columns are present"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    required_columns = [
        'feedback_id', 'original_text',
        'nltk_tokens_count', 'treebank_tokens_count', 'spacy_tokens_count',
        'nltk_sample_tokens', 'treebank_sample_tokens', 'spacy_sample_tokens',
        'nltk_sample_stems', 'treebank_sample_stems', 'spacy_sample_stems'
    ]
    missing = [col for col in required_columns if col not in result.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_token_counts_validity():
    """Test 4: Verify token counts are positive integers"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    count_columns = ['nltk_tokens_count', 'treebank_tokens_count', 'spacy_tokens_count']

    for col in count_columns:
        assert (result[col] > 0).all(), f"Column '{col}' has non-positive values"
        assert pd.api.types.is_integer_dtype(result[col]), f"Column '{col}' is not integer type"


def test_tokens_are_lists():
    """Test 5: Verify token samples are proper lists (can be split)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    token_columns = ['nltk_sample_tokens', 'treebank_sample_tokens', 'spacy_sample_tokens']

    for col in token_columns:
        for idx, value in enumerate(result[col]):
            assert isinstance(value, str) and len(value) > 0, f"Row {idx} in '{col}' is empty or invalid"
            tokens = value.split()
            assert len(tokens) > 0, f"Row {idx} in '{col}' contains no tokens when split"


def test_stemming_applied():
    """Test 6: Verify stems exist and are strings"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    stem_columns = ['nltk_sample_stems', 'treebank_sample_stems', 'spacy_sample_stems']

    for col in stem_columns:
        assert (result[col].str.len() > 0).all(), f"Column '{col}' has empty values"
        assert pd.api.types.is_string_dtype(result[col]) or result[col].dtype == 'object', \
            f"Column '{col}' is not string type"


def test_no_null_values():
    """Test 7: Verify no NULL/NaN values in result"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "customer_feedback.csv")
    output_path = os.path.join(script_dir, "outputs", "test_results.csv")

    result = analyze_tokenization(input_path, output_path)
    null_counts = result.isnull().sum()
    assert null_counts.sum() == 0, f"Found NULL values in columns: {null_counts[null_counts > 0].to_dict()}"


def test_with_mock_data():
    """Test 8: Test function with mock data and verify it returns DataFrame"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create temporary mock data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=script_dir) as mock_input:
        writer = csv.writer(mock_input)
        writer.writerow(['customer_id', 'feedback_text'])
        writer.writerow([1, 'This is excellent'])
        writer.writerow([2, 'Very poor quality'])
        writer.writerow([3, 'Amazing product'])
        mock_input_path = mock_input.name

    mock_output_path = os.path.join(script_dir, "outputs", "mock_test_output.csv")

    try:
        result = analyze_tokenization(mock_input_path, mock_output_path)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame but got {type(result).__name__}"
    finally:
        if os.path.exists(mock_input_path):
            os.remove(mock_input_path)


if __name__ == "__main__":
    test_output_file_creation()
    test_row_count_match()
    test_required_columns()
    test_token_counts_validity()
    test_tokens_are_lists()
    test_stemming_applied()
    test_no_null_values()
    test_with_mock_data()
