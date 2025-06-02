import re
import numpy as np
from typing import List, Dict, Tuple
import torch
from groq import Groq
import spacy
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    exit(1)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SelfCheckGPT:
    def __init__(self, groq_api_key: str = None):
        """Initialize SelfCheckGPT with M1-optimized models."""
        self.client = Groq(api_key=groq_api_key) if groq_api_key else Groq()
        self.device = get_device()
        
        print(f"Using device: {self.device}")
        print("Loading models (optimized for Apple Silicon)...")

        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device.type == "mps":
                self.similarity_model = self.similarity_model.to(self.device)
            print("✅ Sentence transformer loaded")
        except Exception as e:
            print(f"⚠️  Could not load sentence transformer: {e}")
            self.similarity_model = None
        
        self.models = [
            "gemma2-9b-it",
            "meta-llama/llama-4-scout-17b-16e-instruct", 
            "mistral-saba-24b"
        ]
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def generate_samples(self, prompt: str, num_samples: int = 3) -> List[str]:
        """Generate multiple samples from different models for consistency checking."""
        samples = []
        
        for i, model in enumerate(self.models[:num_samples]):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide factual information."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.8,
                    max_tokens=1024,
                    top_p=0.9
                )
                samples.append(response.choices[0].message.content.strip())
                print(f"Generated sample {i+1} using {model}")
            except Exception as e:
                print(f"Error with model {model}: {str(e)}")
                continue
                
        return samples
    
    def simple_nli_check(self, sentence: str, context: str) -> float:
        """Simplified NLI check using string matching and keyword analysis."""
        sentence_lower = sentence.lower()
        context_lower = context.lower()
        
        sentence_words = set(sentence_lower.split())
        context_words = set(context_lower.split())
        
        contradiction_patterns = [
            ("is", "is not"), ("was", "was not"), ("can", "cannot"),
            ("will", "will not"), ("has", "has not"), ("does", "does not")
        ]
        
        contradiction_score = 0.0
        
        for pos, neg in contradiction_patterns:
            if pos in sentence_lower and neg in context_lower:
                contradiction_score += 0.3
            if neg in sentence_lower and pos in context_lower:
                contradiction_score += 0.3
        
        overlap = len(sentence_words.intersection(context_words))
        total_words = len(sentence_words.union(context_words))
        
        if total_words > 0:
            overlap_ratio = overlap / total_words
            inconsistency_from_overlap = 1.0 - overlap_ratio
        else:
            inconsistency_from_overlap = 0.5
        
        final_score = min(1.0, contradiction_score + (inconsistency_from_overlap * 0.5))
        return final_score
    
    def selfcheck_nli(self, sentences: List[str], samples: List[str]) -> List[float]:
        """Check consistency using simplified NLI approach."""
        scores = []
        
        for sentence in sentences:
            contradiction_scores = []
            
            for sample in samples:
                score = self.simple_nli_check(sentence, sample)
                contradiction_scores.append(score)
            
            avg_contradiction = np.mean(contradiction_scores)
            scores.append(avg_contradiction)
            
        return scores
    
    def analyze_inconsistency_type(self, sentence: str, samples: List[str]) -> Dict:
        """Analyze what type of inconsistency we're seeing."""
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
        
        for sample in samples:
            sample_lower = sample.lower()
            
            if sentence_lower in sample_lower:
                exact_matches += 1
            
            sample_words = set(sample_lower.split())
            overlap = len(sentence_words.intersection(sample_words))
            if overlap > len(sentence_words) * 0.5:
                partial_matches += 1
            
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
        """Check consistency using semantic similarity (M1 optimized)."""
        if not self.similarity_model:
            print("⚠️  Similarity model not available, using n-gram fallback")
            return self.selfcheck_ngram_consistency(sentences, samples)
        
        scores = []
        
        try:
            sentence_embeddings = self.similarity_model.encode(sentences)
            sample_embeddings = self.similarity_model.encode(samples)
            
            for i, sent_embedding in enumerate(sentence_embeddings):
                similarities = []
                
                for sample_embedding in sample_embeddings:
                    similarity = np.dot(sent_embedding, sample_embedding) / (
                        np.linalg.norm(sent_embedding) * np.linalg.norm(sample_embedding)
                    )
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                inconsistency_score = max(0.0, 1.0 - avg_similarity)
                scores.append(inconsistency_score)
                
        except Exception as e:
            print(f"⚠️  Error in similarity calculation: {e}")
            return self.selfcheck_ngram_consistency(sentences, samples)
            
        return scores
    
    def selfcheck_ngram_consistency(self, sentences: List[str], samples: List[str], n: int = 1) -> List[float]:
        """Check consistency using n-gram overlap."""
        scores = []
        
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            if len(words) < n:
                return set()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        for sentence in sentences:
            sent_ngrams = get_ngrams(sentence, n)
            if not sent_ngrams:
                scores.append(0.5) 
                continue
            
            overlap_ratios = []
            for sample in samples:
                sample_ngrams = get_ngrams(sample, n)
                if not sample_ngrams:
                    overlap_ratios.append(0.0)
                    continue
                
                intersection = sent_ngrams.intersection(sample_ngrams)
                union = sent_ngrams.union(sample_ngrams)
                overlap_ratio = len(intersection) / len(union) if union else 0.0
                overlap_ratios.append(overlap_ratio)
            
            avg_overlap = np.mean(overlap_ratios)
            inconsistency_score = 1.0 - avg_overlap
            scores.append(max(0.0, inconsistency_score))
            
        return scores
    
    def selfcheck_llm_prompting(self, sentences: List[str], samples: List[str]) -> List[float]:
        """Check consistency using LLM prompting (most effective method)."""
        scores = []
        
        for sentence in sentences:
            consistency_scores = []
            
            for sample in samples:
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
                        model="llama-3.1-8b-instant",
                        temperature=0.1,
                        max_tokens=10
                    )
                    
                    answer = response.choices[0].message.content.strip().lower()
                    
                    if "yes" in answer:
                        consistency_scores.append(0.0)
                    elif "no" in answer:
                        consistency_scores.append(1.0)
                    else:
                        consistency_scores.append(0.5)  # Uncertain
                        
                except Exception as e:
                    print(f"Error in LLM prompting: {e}")
                    consistency_scores.append(0.5)
            
            avg_inconsistency = np.mean(consistency_scores)
            scores.append(avg_inconsistency)
            
        return scores
    
    def detect_hallucinations(self, prompt: str, num_samples: int = 3, methods: List[str] = None) -> Dict:
        """
        Main function to detect hallucinations in LLM responses.
        
        Args:
            prompt: The input prompt to generate responses for
            num_samples: Number of sample responses to generate
            methods: List of methods to use ['nli', 'similarity', 'ngram', 'llm_prompt']
        """
        if methods is None:
            methods = ['nli', 'similarity', 'ngram', 'llm_prompt']
        
        print(f"Generating {num_samples} samples for prompt: '{prompt[:50]}...'")
        
        samples = self.generate_samples(prompt, num_samples)
        
        if len(samples) < 2:
            return {"error": "Could not generate enough samples for comparison"}
        
        main_response = samples[0]
        comparison_samples = samples[1:]
        
        print(f"\nMain response: {main_response[:100]}...")
        print(f"Checking against {len(comparison_samples)} other samples\n")
        
        sentences = self.split_into_sentences(main_response)
        
        results = {
            "main_response": main_response,
            "sentences": sentences,
            "num_samples": len(samples),
            "methods_used": methods,
            "scores": {}
        }
        
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
        
        if results["scores"]:
            all_scores = list(results["scores"].values())
            ensemble_scores = np.mean(all_scores, axis=0).tolist()
            results["scores"]["ensemble"] = ensemble_scores
        
        print("Analyzing inconsistency types...")
        results["detailed_analysis"] = []
        for sentence in sentences:
            analysis = self.analyze_inconsistency_type(sentence, comparison_samples)
            results["detailed_analysis"].append(analysis)
        
        return results
    
    def print_analysis(self, results: Dict, threshold: float = 0.5):
        """Print detailed analysis of hallucination detection results."""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("SELFCHECKGPT CONSISTENCY ANALYSIS")
        print("="*80)
        print("⚠️  NOTE: This detects INCONSISTENCY, not factual accuracy!")
        print("   High scores may indicate true facts that models express differently.")
        
        print(f"\nMain Response: {results['main_response'][:200]}...")
        print(f"Number of sentences analyzed: {len(results['sentences'])}")
        print(f"Number of comparison samples: {results['num_samples'] - 1}")
        
        sentences = results['sentences']
        
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
        
        if "ensemble" in results['scores']:
            ensemble_scores = results['scores']['ensemble']
            avg_score = np.mean(ensemble_scores)
            suspicious_count = sum(1 for score in ensemble_scores if score > threshold)
            
            print(f"\n--- OVERALL ASSESSMENT ---")
            print(f"Average inconsistency score: {avg_score:.3f}")
            print(f"Inconsistent sentences: {suspicious_count}/{len(sentences)}")
            
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
            
            print(f"\n💡 REMEMBER:")
            print("   • Consistency ≠ Accuracy")
            print("   • True facts can be expressed inconsistently")
            print("   • Multiple models can share the same wrong information")
            print("   • Use this as a first-pass filter, not definitive fact-checking")

def main():
    print("Initializing SelfCheckGPT...")
    checker = SelfCheckGPT()
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        try:
            results = checker.detect_hallucinations(
                prompt=prompt,
                num_samples=3,
                methods=['nli', 'similarity', 'llm_prompt']
            )
            
            checker.print_analysis(results, threshold=0.5)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()