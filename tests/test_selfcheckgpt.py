import pytest
from src.main import SelfCheckGPT

@pytest.fixture
def checker():
    """Create a SelfCheckGPT instance for testing."""
    return SelfCheckGPT()

def test_initialization(checker):
    """Test that the checker initializes correctly."""
    assert checker is not None
    assert hasattr(checker, 'client')
    assert hasattr(checker, 'device')

def test_split_into_sentences(checker):
    """Test sentence splitting functionality."""
    text = "This is sentence one. This is sentence two! And this is sentence three?"
    sentences = checker.split_into_sentences(text)
    assert len(sentences) == 3
    assert all(isinstance(s, str) for s in sentences)

def test_simple_nli_check(checker):
    """Test the simple NLI check functionality."""
    sentence = "The cat is on the mat."
    context = "The cat is not on the mat."
    score = checker.simple_nli_check(sentence, context)
    assert 0 <= score <= 1
    assert isinstance(score, float)

def test_ngram_consistency(checker):
    """Test n-gram consistency checking."""
    sentences = ["The cat is on the mat.", "The cat sits on the mat."]
    samples = ["The cat is on the mat.", "The cat is not on the mat."]
    scores = checker.selfcheck_ngram_consistency(sentences, samples)
    assert len(scores) == len(sentences)
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.skip(reason="Requires API key and internet connection")
def test_detect_hallucinations(checker):
    """Test the main hallucination detection functionality."""
    prompt = "What is the capital of France?"
    results = checker.detect_hallucinations(
        prompt=prompt,
        num_samples=2,
        methods=['nli', 'similarity']
    )
    assert isinstance(results, dict)
    assert 'main_response' in results
    assert 'sentences' in results
    assert 'scores' in results 