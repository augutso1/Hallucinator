# SelfCheckGPT

A multi-method hallucination detection system for Large Language Models (LLMs) that checks consistency across multiple model outputs.

## Features

- Multiple detection methods:
  - Natural Language Inference (NLI)
  - Semantic Similarity
  - N-gram Consistency
  - LLM Prompting
- Optimized for Apple Silicon (M1/M2) Macs
- Detailed inconsistency analysis
- Ensemble scoring system
- Comprehensive documentation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/SelfCheckGPT.git
cd SelfCheckGPT
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. For M1/M2 Mac users, follow these additional steps:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install sentence-transformers with specific version
pip install --upgrade --no-deps --force-reinstall sentence-transformers==2.2.2

# Install spaCy model
python -m spacy download en_core_web_sm
```

## Usage

```python
from src.main import SelfCheckGPT

# Initialize the checker
checker = SelfCheckGPT(groq_api_key="your_api_key")

# Detect hallucinations in a response
results = checker.detect_hallucinations(
    prompt="Your prompt here",
    num_samples=3,
    methods=['nli', 'similarity', 'llm_prompt']
)

# Print analysis
checker.print_analysis(results, threshold=0.5)
```

## Methods

1. **NLI (Natural Language Inference)**

   - Fast, rule-based approach
   - Detects explicit contradictions
   - No GPU required

2. **Semantic Similarity**

   - Uses sentence transformers
   - Captures semantic relationships
   - Optimized for M1/M2 Macs

3. **N-gram Consistency**

   - Simple but effective baseline
   - Fast computation
   - Good fallback option

4. **LLM Prompting**
   - Most sophisticated method
   - Uses LLM as a judge
   - Best for complex reasoning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the SelfCheckGPT methodology
- Optimized for Apple Silicon architecture
- Uses Groq API for fast inference
