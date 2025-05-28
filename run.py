from bert_score import score as bertscore
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Get API key from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable")

client = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Evidence organized by themes
evidence_by_theme = {
    "brazil": [
        "The capital of Brazil is Brasília.",
        "Brazil is a country in South America.",
        "São Paulo is a large city in Brazil.",
        "The current president of Brazil is Luiz Inácio Lula da Silva, commonly known as Lula.",
        "Lula was elected president of Brazil in 2022.",
        "Brazil is the largest country in South America.",
        "Brazil's official language is Portuguese."
    ],
    "technology": [
        "Apple's M1 chip was released in 2020.",
        "The M1 Pro and M1 Max chips were introduced in 2021.",
        "M1 Max supports up to 64GB of unified memory.",
        "M1 Pro has up to 10 CPU cores and 16 GPU cores.",
        "M1 Max has up to 10 CPU cores and 32 GPU cores.",
        "Both chips are built on 5-nanometer process technology."
    ],
    "science": [
        "The speed of light is approximately 299,792,458 meters per second.",
        "Water's chemical formula is H2O.",
        "The Earth's core is primarily made of iron and nickel.",
        "DNA stands for deoxyribonucleic acid.",
        "The human body contains about 60% water.",
        "Photosynthesis is the process plants use to convert light into energy."
    ]
}

def generate(prompt: str, num_samples: int = 3):
    samples = []
    for _ in range(num_samples):
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,  # Slightly lower temperature for more focused responses
            max_tokens=100,
            top_p=1,
            stream=False
        )
        samples.append(completion.choices[0].message.content)
    return samples

def get_relevant_evidence(prompt: str, evidence_dict: dict) -> list:
    # Encode the prompt
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    
    # Calculate similarity with each theme
    theme_similarities = {}
    for theme, evidence in evidence_dict.items():
        theme_text = " ".join(evidence)
        theme_embedding = model.encode(theme_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(prompt_embedding, theme_embedding)[0][0]
        theme_similarities[theme] = float(similarity)
    
    # Get the most relevant theme
    most_relevant_theme = max(theme_similarities.items(), key=lambda x: x[1])[0]
    return evidence_dict[most_relevant_theme]

def score_sentences(sentences, evidence):
    scores = []
    # Encode all sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    evidence_embeddings = model.encode(evidence, convert_to_tensor=True)
    
    # Calculate cosine similarity
    for i, sent_emb in enumerate(sentence_embeddings):
        similarities = util.pytorch_cos_sim(sent_emb, evidence_embeddings)[0]
        best_score = float(torch.max(similarities))
        best_evidence_idx = int(torch.argmax(similarities))
        best_evidence = evidence[best_evidence_idx]
        scores.append((sentences[i], best_score, best_evidence))
    return scores

def check_consistency(samples: list) -> float:
    # Encode all samples
    sample_embeddings = model.encode(samples, convert_to_tensor=True)
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sim = util.pytorch_cos_sim(sample_embeddings[i], sample_embeddings[j])[0][0]
            similarities.append(float(sim))
    
    # Return average similarity
    return sum(similarities) / len(similarities) if similarities else 0.0

if __name__ == "__main__":
    prompt = input("Your prompt: ")
    samples = generate(prompt)
    
    print("\nGenerated Samples:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        sentences = sent_tokenize(sample)
        for s in sentences:
            print(f"• {s}")
    
    # Get relevant evidence based on the prompt
    relevant_evidence = get_relevant_evidence(prompt, evidence_by_theme)
    
    print("\nConsistency Check:")
    consistency_score = check_consistency(samples)
    print(f"Average similarity between samples: {consistency_score:.2f}")
    
    print("\nScoring First Sample Against Evidence:")
    scored = score_sentences(sent_tokenize(samples[0]), relevant_evidence)
    for sent, score, best_evidence in scored:
        print(f"\nSentence: {sent}")
        print(f"Score: {score:.2f}")
        print(f"Most similar evidence: {best_evidence}")