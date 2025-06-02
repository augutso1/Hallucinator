"""
SelfCheckGPT: A Multi-Method Hallucination Detection System

This implementation uses the SelfCheckGPT methodology to detect potential hallucinations
in Large Language Model (LLM) responses by checking consistency across multiple model outputs.

Key Principle: If an LLM hallucinates, the hallucinated content is likely to be inconsistent
when the same prompt is given to multiple models or the same model multiple times.

Architecture Overview:
1. Generate multiple responses from different LLMs for the same prompt
2. Compare the main response against other samples using various consistency metrics
3. Identify sentences that show high inconsistency (potential hallucinations)
4. Provide detailed analysis of inconsistency types and confidence levels

Hardware Optimization: Specifically optimized for Apple Silicon (M1/M2) Macs with
fallback support for CUDA and CPU execution.
"""

import re
import numpy as np
from typing import List, Dict, Tuple
import torch
from groq import Groq
import spacy
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings to keep output clean during model loading and inference
warnings.filterwarnings("ignore")

# Load spaCy model for sentence segmentation
# We use spaCy over simple punctuation splitting because it handles edge cases better
# (abbreviations, decimal numbers, etc.)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    exit(1)

def get_device():
    """
    Determine the best available device for PyTorch computations.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - Apple Silicon GPU acceleration
    2. CUDA - NVIDIA GPU acceleration  
    3. CPU - Fallback for compatibility
    
    This optimization is crucial for sentence transformer performance on M1/M2 Macs.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SelfCheckGPT:
    def __init__(self, groq_api_key: str = None):
        """
        Initialize SelfCheckGPT with M1-optimized models.
        
        Design Decision: Using Groq API for fast inference with multiple models
        - Groq provides high-speed inference for various open-source models
        - Multiple models increase diversity in responses, improving detection accuracy
        - Alternative to running multiple local models (memory/compute intensive)
        """
        self.client = Groq(api_key=groq_api_key) if groq_api_key else Groq()
        self.device = get_device()
        
        print(f"Using device: {self.device}")
        print("Loading models (optimized for Apple Silicon)...")

        # Load sentence transformer for semantic similarity calculations
        # Model choice: 'all-MiniLM-L6-v2' provides good balance of speed vs accuracy
        # Small enough to run efficiently on consumer hardware
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Move to appropriate device for optimal performance
            if self.device.type == "mps":
                self.similarity_model = self.similarity_model.to(self.device)
            print("✅ Sentence transformer loaded")
        except Exception as e:
            print(f"⚠️  Could not load sentence transformer: {e}")
            self.similarity_model = None
        
        # Model Selection Strategy:
        # Using diverse model families to maximize response variation
        # - Gemma2: Google's efficient model
        # - LLama-4: Meta's latest reasoning-focused model  
        # - Mistral: European model with different training approach
        # Diversity in training data/methods increases hallucination detection sensitivity
        self.models = [
            "gemma2-9b-it",
            "meta-llama/llama-4-scout-17b-16e-instruct", 
            "mistral-saba-24b"
        ]
        
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy's linguistic analysis.
        
        Why spaCy over regex/simple splitting:
        - Handles abbreviations (Dr., etc.) correctly
        - Manages decimal numbers (3.14 is not sentence boundary)
        - Considers context for ambiguous punctuation
        - Better handling of quotations and complex punctuation
        
        This granular sentence-level analysis is crucial for SelfCheckGPT
        as hallucinations often occur at the sentence level.
        """
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def generate_samples(self, prompt: str, num_samples: int = 3) -> List[str]:
        """
        Generate multiple samples from different models for consistency checking.
        
        Methodology Explanation:
        - Use different models rather than same model with different temperatures
        - Each model has different training data, architectures, and biases
        - True facts tend to be consistent across different models
        - Hallucinations are more likely to be model-specific and inconsistent
        
        Temperature Setting (0.8): Balanced approach
        - High enough to allow natural variation in expression
        - Low enough to maintain factual consistency
        - Prevents overly deterministic responses that might mask hallucinations
        """
        samples = []
        
        for i, model in enumerate(self.models[:num_samples]):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide factual information."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.8,  # Balanced: allows variation while maintaining coherence
                    max_tokens=1024,  # Sufficient for detailed responses
                    top_p=0.9         # Nucleus sampling for quality control
                )
                samples.append(response.choices[0].message.content.strip())
                print(f"Generated sample {i+1} using {model}")
            except Exception as e:
                # Graceful degradation: Continue with available models if some fail
                print(f"Error with model {model}: {str(e)}")
                continue
                
        return samples
    
    def simple_nli_check(self, sentence: str, context: str) -> float:
        """
        Simplified Natural Language Inference (NLI) check using pattern matching.
        
        Design Decision: Rule-based approach over transformer-based NLI
        Reasoning:
        - Faster inference (no GPU required for this component)
        - More interpretable results
        - Sufficient accuracy for initial consistency screening
        - Reduces dependency on additional large models
        
        Method combines two approaches:
        1. Contradiction Pattern Detection: Explicit negation patterns
        2. Word Overlap Analysis: Semantic similarity proxy
        
        Score Interpretation: 0.0 = perfectly consistent, 1.0 = completely contradictory
        """
        sentence_lower = sentence.lower()
        context_lower = context.lower()
        
        # Convert to word sets for overlap analysis
        sentence_words = set(sentence_lower.split())
        context_words = set(context_lower.split())
        
        # Explicit contradiction patterns
        # These patterns catch direct logical contradictions
        contradiction_patterns = [
            ("is", "is not"), ("was", "was not"), ("can", "cannot"),
            ("will", "will not"), ("has", "has not"), ("does", "does not")
        ]
        
        contradiction_score = 0.0
        
        # Check for explicit contradictions in both directions
        for pos, neg in contradiction_patterns:
            if pos in sentence_lower and neg in context_lower:
                contradiction_score += 0.3  # Weight chosen empirically
            if neg in sentence_lower and pos in context_lower:
                contradiction_score += 0.3
        
        # Word overlap analysis
        # Low overlap suggests different information (potential inconsistency)
        overlap = len(sentence_words.intersection(context_words))
        total_words = len(sentence_words.union(context_words))
        
        if total_words > 0:
            overlap_ratio = overlap / total_words
            inconsistency_from_overlap = 1.0 - overlap_ratio
        else:
            # Edge case: empty sentences
            inconsistency_from_overlap = 0.5
        
        # Combine scores with weights
        # Contradiction patterns get higher weight as they're more reliable indicators
        final_score = min(1.0, contradiction_score + (inconsistency_from_overlap * 0.5))
        return final_score
    
    def selfcheck_nli(self, sentences: List[str], samples: List[str]) -> List[float]:
        """
        Check consistency using simplified NLI approach.
        
        Implementation Strategy:
        - Compare each sentence against all comparison samples
        - Average contradiction scores across samples
        - Higher scores indicate potential hallucinations
        
        This method is the core of SelfCheckGPT, implementing the paper's
        primary methodology with efficiency optimizations.
        """
        scores = []
        
        for sentence in sentences:
            contradiction_scores = []
            
            # Compare sentence against each sample
            for sample in samples:
                score = self.simple_nli_check(sentence, sample)
                contradiction_scores.append(score)
            
            # Average across all comparisons for robustness
            avg_contradiction = np.mean(contradiction_scores)
            scores.append(avg_contradiction)
            
        return scores
    
    def analyze_inconsistency_type(self, sentence: str, samples: List[str]) -> Dict:
        """
        Analyze what type of inconsistency we're seeing.
        
        Inconsistency Classification System:
        1. likely_consistent: High agreement across models
        2. expression_variation: Same facts, different wording
        3. direct_contradiction: Explicit disagreement
        4. unique_information: Info only in main response
        5. mixed_signals: Unclear pattern requiring manual review
        
        This classification helps users understand the nature of potential
        hallucinations and prioritize manual fact-checking efforts.
        """
        analysis = {
            "sentence": sentence,
            "inconsistency_type": "unknown",
            "explanation": "",
            "confidence": "low"
        }
        
        exact_matches = 0
        partial_matches = 0
        contradictions = 0
        
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        # Analyze each sample for different types of consistency/inconsistency
        for sample in samples:
            sample_lower = sample.lower()
            
            # Exact match: sentence appears verbatim in sample
            if sentence_lower in sample_lower:
                exact_matches += 1
            
            # Partial match: significant word overlap (>50% of sentence words)
            sample_words = set(sample_lower.split())
            overlap = len(sentence_words.intersection(sample_words))
            if overlap > len(sentence_words) * 0.5:
                partial_matches += 1
            
            # Direct contradiction: explicit negation patterns
            contradiction_patterns = [
                ("is", "is not"), ("was", "was not"), ("can", "cannot"),
                ("will", "will not"), ("has", "has not"), ("true", "false")
            ]
            
            for pos, neg in contradiction_patterns:
                if (pos in sentence_lower and neg in sample_lower) or \
                   (neg in sentence_lower and pos in sample_lower):
                    contradictions += 1
                    break
        
        total_samples = len(samples)
        
        # Classification logic with empirically determined thresholds
        if exact_matches >= total_samples * 0.8:
            analysis["inconsistency_type"] = "likely_consistent"
            analysis["explanation"] = "Sentence appears consistently across most samples"
            analysis["confidence"] = "high"
        elif partial_matches >= total_samples * 0.6:
            analysis["inconsistency_type"] = "expression_variation"
            analysis["explanation"] = "Same information expressed differently across models"
            analysis["confidence"] = "medium"
        elif contradictions > 0:
            analysis["inconsistency_type"] = "direct_contradiction"
            analysis["explanation"] = "Models directly contradict each other on this point"
            analysis["confidence"] = "high"
        elif partial_matches == 0 and exact_matches == 0:
            analysis["inconsistency_type"] = "unique_information" 
            analysis["explanation"] = "Information appears only in main response, not in other samples"
            analysis["confidence"] = "medium"
        else:
            analysis["inconsistency_type"] = "mixed_signals"
            analysis["explanation"] = "Unclear pattern - requires manual review"
            analysis["confidence"] = "low"
        
        return analysis
    
    def selfcheck_semantic_similarity(self, sentences: List[str], samples: List[str]) -> List[float]:
        """
        Check consistency using semantic similarity (M1 optimized).
        
        Method Explanation:
        - Uses sentence transformers to create dense vector representations
        - Computes cosine similarity between sentence and sample embeddings
        - More sophisticated than word overlap, captures semantic meaning
        
        Optimization Notes:
        - Model moved to MPS device for Apple Silicon acceleration
        - Graceful fallback to n-gram method if transformer fails
        - Batch processing for efficiency
        
        This method complements rule-based NLI by capturing semantic
        relationships that might be missed by simple pattern matching.
        """
        if not self.similarity_model:
            print("⚠️  Similarity model not available, using n-gram fallback")
            return self.selfcheck_ngram_consistency(sentences, samples)
        
        scores = []
        
        try:
            # Batch encode for efficiency (especially important on GPU)
            sentence_embeddings = self.similarity_model.encode(sentences)
            sample_embeddings = self.similarity_model.encode(samples)
            
            for i, sent_embedding in enumerate(sentence_embeddings):
                similarities = []
                
                # Compare sentence embedding to each sample embedding
                for sample_embedding in sample_embeddings:
                    # Cosine similarity calculation
                    similarity = np.dot(sent_embedding, sample_embedding) / (
                        np.linalg.norm(sent_embedding) * np.linalg.norm(sample_embedding)
                    )
                    similarities.append(similarity)
                
                # Convert similarity to inconsistency score
                avg_similarity = np.mean(similarities)
                inconsistency_score = max(0.0, 1.0 - avg_similarity)
                scores.append(inconsistency_score)
                
        except Exception as e:
            # Graceful degradation to simpler method
            print(f"⚠️  Error in similarity calculation: {e}")
            return self.selfcheck_ngram_consistency(sentences, samples)
            
        return scores
    
    def selfcheck_ngram_consistency(self, sentences: List[str], samples: List[str], n: int = 1) -> List[float]:
        """
        Check consistency using n-gram overlap.
        
        N-gram Method Explanation:
        - Breaks text into sequences of N words (unigrams when n=1)
        - Computes Jaccard similarity (intersection over union)
        - Fast, simple method that works well as a baseline
        
        Choice of n=1 (unigrams):
        - More robust to word order variations
        - Sufficient for content overlap detection
        - Faster computation than higher-order n-grams
        
        This serves as a reliable fallback when transformer models fail
        and provides a different perspective on textual similarity.
        """
        scores = []
        
        def get_ngrams(text: str, n: int) -> set:
            """Extract n-grams from text."""
            words = text.lower().split()
            if len(words) < n:
                return set()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        for sentence in sentences:
            sent_ngrams = get_ngrams(sentence, n)
            if not sent_ngrams:
                # Handle edge case of very short sentences
                scores.append(0.5) 
                continue
            
            overlap_ratios = []
            for sample in samples:
                sample_ngrams = get_ngrams(sample, n)
                if not sample_ngrams:
                    overlap_ratios.append(0.0)
                    continue
                
                # Jaccard similarity: intersection over union
                intersection = sent_ngrams.intersection(sample_ngrams)
                union = sent_ngrams.union(sample_ngrams)
                overlap_ratio = len(intersection) / len(union) if union else 0.0
                overlap_ratios.append(overlap_ratio)
            
            # Convert overlap to inconsistency score
            avg_overlap = np.mean(overlap_ratios)
            inconsistency_score = 1.0 - avg_overlap
            scores.append(max(0.0, inconsistency_score))
            
        return scores
    
    def selfcheck_llm_prompting(self, sentences: List[str], samples: List[str]) -> List[float]:
        """
        Check consistency using LLM prompting (most effective method).
        
        Why This Method is Most Effective:
        - Leverages the reasoning capabilities of LLMs for nuanced evaluation
        - Can understand context and implicit relationships
        - More sophisticated than rule-based or similarity approaches
        - Handles complex logical relationships and domain-specific knowledge
        
        Implementation Details:
        - Uses a lightweight, fast model (llama-3.1-8b-instant) for efficiency
        - Low temperature (0.1) for consistent, deterministic judgments
        - Binary classification (Yes/No) for clear decision boundaries
        - Short max_tokens to force concise answers
        
        This method essentially uses an LLM as a judge to evaluate
        consistency, which often aligns well with human judgment.
        """
        scores = []
        
        for sentence in sentences:
            consistency_scores = []
            
            for sample in samples:
                # Carefully crafted prompt for consistent evaluation
                prompt = f"""Context: {sample}

Sentence: {sentence}

Question: Is the sentence above supported by the context? Consider if the facts, claims, and information in the sentence are consistent with what's stated in the context.

Answer with only 'Yes' or 'No'."""

                try:
                    response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a fact-checker. Analyze whether sentences are supported by given context."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.1-8b-instant",  # Fast, efficient model for this task
                        temperature=0.1,  # Low temperature for consistent judgments
                        max_tokens=10     # Force concise binary answers
                    )
                    
                    answer = response.choices[0].message.content.strip().lower()
                    
                    # Convert text response to numerical score
                    if "yes" in answer:
                        consistency_scores.append(0.0)  # Consistent = low inconsistency
                    elif "no" in answer:
                        consistency_scores.append(1.0)  # Inconsistent = high inconsistency
                    else:
                        consistency_scores.append(0.5)  # Uncertain = medium inconsistency
                        
                except Exception as e:
                    print(f"Error in LLM prompting: {e}")
                    consistency_scores.append(0.5)  # Default to uncertain on error
            
            # Average inconsistency across all samples
            avg_inconsistency = np.mean(consistency_scores)
            scores.append(avg_inconsistency)
            
        return scores
    
    def detect_hallucinations(self, prompt: str, num_samples: int = 3, methods: List[str] = None) -> Dict:
        """
        Main function to detect hallucinations in LLM responses.
        
        Multi-Method Ensemble Approach:
        - Combines multiple detection methods for robustness
        - Each method has different strengths and blind spots
        - Ensemble averaging reduces false positives/negatives
        
        Method Selection Rationale:
        - NLI: Fast, interpretable, good for explicit contradictions
        - Similarity: Captures semantic relationships, handles paraphrasing
        - N-gram: Simple baseline, robust fallback
        - LLM Prompting: Most sophisticated, best for complex reasoning
        
        Args:
            prompt: The input prompt to generate responses for
            num_samples: Number of sample responses to generate (default 3)
            methods: List of methods to use ['nli', 'similarity', 'ngram', 'llm_prompt']
        
        Returns:
            Comprehensive results dictionary with scores, analysis, and metadata
        """
        if methods is None:
            # Default to most effective methods, balanced for speed vs accuracy
            methods = ['nli', 'similarity', 'ngram', 'llm_prompt']
        
        print(f"Generating {num_samples} samples for prompt: '{prompt[:50]}...'")
        
        # Generate diverse responses for comparison
        samples = self.generate_samples(prompt, num_samples)
        
        if len(samples) < 2:
            return {"error": "Could not generate enough samples for comparison"}
        
        # Use first sample as main response, others for comparison
        # This design allows for easy extension to user-provided responses
        main_response = samples[0]
        comparison_samples = samples[1:]
        
        print(f"\nMain response: {main_response[:100]}...")
        print(f"Checking against {len(comparison_samples)} other samples\n")
        
        # Sentence-level analysis (core of SelfCheckGPT methodology)
        sentences = self.split_into_sentences(main_response)
        
        results = {
            "main_response": main_response,
            "sentences": sentences,
            "num_samples": len(samples),
            "methods_used": methods,
            "scores": {}
        }
        
        # Apply each selected method
        if 'nli' in methods:
            print("Running NLI consistency check...")
            nli_scores = self.selfcheck_nli(sentences, comparison_samples)
            results["scores"]["nli"] = nli_scores
        
        if 'similarity' in methods:
            print("Running semantic similarity check...")
            sim_scores = self.selfcheck_semantic_similarity(sentences, comparison_samples)
            results["scores"]["similarity"] = sim_scores
        
        if 'ngram' in methods:
            print("Running n-gram consistency check...")
            ngram_scores = self.selfcheck_ngram_consistency(sentences, comparison_samples)
            results["scores"]["ngram"] = ngram_scores
        
        if 'llm_prompt' in methods:
            print("Running LLM prompting check...")
            llm_scores = self.selfcheck_llm_prompting(sentences, comparison_samples)
            results["scores"]["llm_prompt"] = llm_scores
        
        # Create ensemble scores by averaging across methods
        # This reduces variance and improves overall detection accuracy
        if results["scores"]:
            all_scores = list(results["scores"].values())
            ensemble_scores = np.mean(all_scores, axis=0).tolist()
            results["scores"]["ensemble"] = ensemble_scores
        
        # Detailed analysis for interpretability
        print("Analyzing inconsistency types...")
        results["detailed_analysis"] = []
        for sentence in sentences:
            analysis = self.analyze_inconsistency_type(sentence, comparison_samples)
            results["detailed_analysis"].append(analysis)
        
        return results
    
    def print_analysis(self, results: Dict, threshold: float = 0.5):
        """
        Print detailed analysis of hallucination detection results.
        
        Presentation Strategy:
        - Clear visual indicators (emojis) for quick assessment
        - Detailed explanations for educational value
        - Explicit warnings about limitations
        - Actionable guidance for users
        
        Threshold Selection (0.5):
        - Empirically determined balance point
        - Adjustable based on use case (higher for more conservative detection)
        - Users can modify based on their risk tolerance
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("SELFCHECKGPT CONSISTENCY ANALYSIS")
        print("="*80)
        # Important disclaimer about methodology limitations
        print("⚠️  NOTE: This detects INCONSISTENCY, not factual accuracy!")
        print("   High scores may indicate true facts that models express differently.")
        
        print(f"\nMain Response: {results['main_response'][:200]}...")
        print(f"Number of sentences analyzed: {len(results['sentences'])}")
        print(f"Number of comparison samples: {results['num_samples'] - 1}")
        
        sentences = results['sentences']
        
        # Method-by-method breakdown for detailed analysis
        for method, scores in results['scores'].items():
            print(f"\n--- {method.upper()} METHOD ---")
            
            for i, (sentence, score) in enumerate(zip(sentences, scores)):
                if score > threshold:
                    status = "🔴 INCONSISTENT ACROSS MODELS"
                    interpretation = "Models disagree on this - could be hallucination OR true but differently expressed"
                else:
                    status = "🟢 CONSISTENT ACROSS MODELS"
                    interpretation = "Models agree on this information"
                
                print(f"\nSentence {i+1}: {sentence}")
                print(f"Score: {score:.3f} ({status})")
                print(f"Interpretation: {interpretation}")
        
        # Overall assessment with actionable guidance
        if "ensemble" in results['scores']:
            ensemble_scores = results['scores']['ensemble']
            avg_score = np.mean(ensemble_scores)
            suspicious_count = sum(1 for score in ensemble_scores if score > threshold)
            
            print(f"\n--- OVERALL ASSESSMENT ---")
            print(f"Average inconsistency score: {avg_score:.3f}")
            print(f"Inconsistent sentences: {suspicious_count}/{len(sentences)}")
            
            # Risk-stratified interpretation with clear action items
            print(f"\n📊 INTERPRETATION GUIDE:")
            if avg_score > 0.7:
                print("🔴 HIGH INCONSISTENCY: Models strongly disagree")
                print("   → Could indicate hallucinations OR complex/nuanced true facts")
                print("   → Requires manual fact-checking for verification")
            elif avg_score > 0.4:
                print("🟡 MODERATE INCONSISTENCY: Some disagreement between models")
                print("   → Mix of consistent and inconsistent information")
                print("   → Check individual sentences marked as inconsistent")
            else:
                print("🟢 LOW INCONSISTENCY: Models mostly agree")
                print("   → Information appears consistent across different sources")
                print("   → Higher confidence in response reliability")
            
            # Important reminders about methodology limitations
            print(f"\n💡 REMEMBER:")
            print("   • Consistency ≠ Accuracy")
            print("   • True facts can be expressed inconsistently")
            print("   • Multiple models can share the same wrong information")
            print("   • Use this as a first-pass filter, not definitive fact-checking")

def main():
    """
    Interactive main function for testing the hallucination detection system.
    
    Design Features:
    - Simple command-line interface for easy testing
    - Graceful error handling with detailed error messages
    - Configurable methods and parameters
    - Clear exit conditions
    
    Default Configuration:
    - 3 samples for good balance of speed vs accuracy
    - Selected methods provide comprehensive coverage
    - 0.5 threshold balances sensitivity vs specificity
    """
    print("Initializing SelfCheckGPT...")
    checker = SelfCheckGPT()
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        try:
            # Run detection with optimized default settings
            results = checker.detect_hallucinations(
                prompt=prompt,
                num_samples=3,  # Good balance of accuracy vs API cost
                methods=['nli', 'similarity', 'llm_prompt']  # Most effective combination
            )
            
            checker.print_analysis(results, threshold=0.5)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()