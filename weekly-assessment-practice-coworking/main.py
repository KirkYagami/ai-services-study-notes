"""Practice: implement coworking handbook question answering with section-attributed RAG."""

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP
MIN_CHUNK_LENGTH = 50
N_RESULTS = 3


def load_and_chunk_document(document_path):
    """Extract sections and split their bodies into overlapping windows."""
    # TODO: implement this function.
    raise NotImplementedError("Implement load_and_chunk_document")


def retrieve_sections(question, embedding_model, collection):
    """Retrieve the four closest chunks and retain their section labels."""
    # TODO: implement this function.
    raise NotImplementedError("Implement retrieve_sections")


def augment_prompt(question, retrieved_chunks):
    """Build a grounded prompt with numbered, section-labelled excerpts."""
    # TODO: implement this function.
    raise NotImplementedError("Implement augment_prompt")


def generate_answer(prompt, client):
    """Send the prompt through the supplied Gemini client."""
    # TODO: implement this function.
    raise NotImplementedError("Implement generate_answer")


def coworking_qa_pipeline(question, document_path):
    """Index the handbook, retrieve context, and generate an attributed answer."""
    # TODO: implement this function.
    raise NotImplementedError("Implement coworking_qa_pipeline")


if __name__ == "__main__":
    # TODO: read and strip the question; reject empty input.
    # TODO: run the pipeline and print question, retrieval scores, sources and answer.
    pass
