!pip install transformers sentence-transformers torch accelerate wikipedia

import re
import json
import torch
import wikipedia
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Initialize Models -----------------
# Paraphraser Model (T5-based)
paraphrase_model_name = "humarin/chatgpt_paraphraser_on_T5_base"
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name).to(device)

# QA Model (Falcon)
qa_model_name = "tiiuae/falcon-7b-instruct"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForCausalLM.from_pretrained(
    qa_model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
qa_tokenizer.pad_token = qa_tokenizer.eos_token  # Set pad token to avoid warnings

# Sentence Similarity Model (SBERT)
sbert_model = SentenceTransformer('all-mpnet-base-v2', device=str(device))

# ----------------- Helper Functions -----------------
def clean_text(text):
    """
    Clean text by removing extra spaces and special prompt artifacts.
    """
    text = re.sub(r'\b(Answer the following question:|Answer:|Question:)\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def clean_answer(answer):
    """
    Remove lines containing 'Answer the following question:'
    to extract only the factual statement.
    """
    lines = answer.split("\n")
    cleaned_lines = [line.strip() for line in lines if "Answer the following question:" not in line]
    return " ".join(cleaned_lines).strip()

def reconstruct_claim(question, answer):
    """
    Reconstructs the claim in a meaningful format.
    If the answer is short (like "Au"), it combines with the question.
    """
    if len(answer.split()) <= 10:  # If answer is too short
        return f"{question} {answer}"
    return answer

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts using SBERT."""
    embeddings = sbert_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()


def get_wikipedia_content(claim, max_chars=5000):
    """
    Fetch Wikipedia content relevant to the claim.
    Try several search queries and return content from the first successful page.
    """
    try:
        # Remove numbers from claim for a cleaner search query if needed
        search_query = re.sub(r'\b\d+\b', '', claim).strip()
        queries = [search_query, claim.split(":")[0]]
        for query in queries:
            try:
                page = wikipedia.page(query, auto_suggest=True)
                content = page.content
                # Return up to max_chars characters of the content
                return content[:max_chars]
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
        # Fallback: use a generic page if no match found
        return wikipedia.summary("Solar System", sentences=10)
    except Exception as e:
        print(f"Wikipedia error: {str(e)}")
        return None

def find_best_match(claim, content):
    """
    Split the content into chunks (sentences) and compute SBERT similarity.
    Return the best matching chunk, its similarity, and the list of all similarities.
    """
    # A simple split using periods and newlines
    chunks = [chunk.strip() for chunk in re.split(r'\n+|\. ', content) if chunk.strip()]
    if not chunks:
        return {"best_chunk": "", "similarity": 0.0, "all_similarities": []}

    claim_embed = sbert_model.encode(claim, convert_to_tensor=True)
    chunk_embeds = sbert_model.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(claim_embed, chunk_embeds)[0]
    top_idx = similarities.argmax().item()
    return {
        "best_chunk": chunks[top_idx],
        "similarity": similarities[top_idx].item(),
        "all_similarities": similarities.tolist()
    }

# ----------------- QA and Paraphrase Functions -----------------
def generate_paraphrases(question, num_paraphrases=5):
    """Generate paraphrased questions using the T5 paraphraser."""
    inputs = paraphrase_tokenizer(
        [f"paraphrase: {question}"],
        return_tensors="pt",
        padding=True,
        max_length=128,
        truncation=True
    ).to(device)

    outputs = paraphrase_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_return_sequences=num_paraphrases,
        num_beams=5,
        max_length=128,
        early_stopping=True,
        repetition_penalty=2.5,
        pad_token_id=paraphrase_tokenizer.eos_token_id
    )
    paraphrases = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return list(set(paraphrases))  # Remove duplicates

def get_answer(question):
    """Query the Falcon model to get an answer, then clean it."""
    input_text = f"Answer the following question: {question}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = qa_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=qa_tokenizer.eos_token_id
    )
    raw_answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_answer(raw_answer)

def verify_fact(question, answer, similarity_threshold=0.65):
    """
    Verify a claim using Wikipedia content.
    If the answer is too short, it reconstructs a claim using the seed question.
    """
    reconstructed_claim = reconstruct_claim(question, clean_answer(answer))

    result = {
        "claim": reconstructed_claim,
        "verified": False,
        "similarity": 0.0,
        "explanation": "",
        "best_match": "",
        "threshold": similarity_threshold
    }
    try:
        wiki_content = get_wikipedia_content(reconstructed_claim)
        if not wiki_content:
            result["explanation"] = "No relevant Wikipedia content found."
            return result

        match_result = find_best_match(reconstructed_claim, wiki_content)
        result["similarity"] = match_result["similarity"]
        result["best_match"] = match_result["best_chunk"]

        if result["similarity"] >= similarity_threshold:
            result["verified"] = True
            result["explanation"] = f"Matches Wikipedia content (similarity: {result['similarity']:.2f})."
        else:
            result["explanation"] = (f"Does not match Wikipedia content (similarity: {result['similarity']:.2f}). "
                                     f"Best match excerpt: {result['best_match'][:200]}...")
    except Exception as e:
        result["explanation"] = f"Verification error: {str(e)}"

    return result

def check_consistency(similarities, pass_threshold=0.6):
    """Check if at least 4 out of 5 paraphrased responses are consistent."""
    consistent_count = sum(1 for sim in similarities if sim >= pass_threshold)

    passed = consistent_count >= 4
    return passed, consistent_count

# ----------------- Main Workflow -----------------
def main():
    seed_question = "Who painted the Mona Lisa?"
    print("Seed Question: ", seed_question)
    # Generate paraphrased questions
    paraphrases = generate_paraphrases(seed_question, num_paraphrases=5)
    print("\nGenerated Paraphrases:")
    for i, para in enumerate(paraphrases, 1):
        print(f"{i}. {para}")

    # Get the answer for the seed question
    seed_answer = get_answer(seed_question)
    print(f"\nSeed Answer: {seed_answer}")

    # Get answers for each paraphrased question
    para_answers = [get_answer(para) for para in paraphrases]
    print("\nParaphrase Answers:")
    for i, (para, ans) in enumerate(zip(paraphrases, para_answers), 1):
        print(f"{i}. {para} : {ans}")

    # Calculate logical consistency similarity scores
    similarities = [compute_similarity(seed_answer, ans) for ans in para_answers]
    consistency_threshold = 0.6  # Adjust based on your requirements
    passed, consistent_count = check_consistency(similarities, pass_threshold=consistency_threshold)

    print("\nLogical Consistency Similarity Scores:")
    for i, sim in enumerate(similarities, 1):
        print(f"Paraphrase {i}: {sim:.2f}")

    print(f"\nLogical Consistency Check (Threshold = {consistency_threshold}): {'Passed' if passed else 'Failed'} ({consistent_count}/5 consistent responses)")

    # Fact Verification using Wikipedia on the seed answer
    fact_verification_result = verify_fact(seed_question, seed_answer, similarity_threshold=0.6)
    print("\nFact Verification Result:")
    print(json.dumps(fact_verification_result, indent=2))

if __name__ == "__main__":
    main()