import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Step 1: Load the Quora Question Pairs dataset
def prepare_quora_dataset(file_path):
    # Load dataset from a CSV file
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.2, random_state=42)
    # Ensure the dataset has the required columns
    if not {"question1", "question2"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'question1' and 'question2' columns.")
    return df

# Step 2: Load BERT model and tokenizer
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

# Step 3: Encode text using BERT
def encode_text(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    # Use the [CLS] token representation
    return outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()

# Step 4: Compute similarity scores for question pairs
def compute_similarity(df, tokenizer, model):
    question1_embeddings = []
    question2_embeddings = []

    for question1, question2 in zip(df["question1"], df["question2"]):
        question1_embeddings.append(encode_text(str(question1), tokenizer, model))
        question2_embeddings.append(encode_text(str(question2), tokenizer, model))

    # Compute cosine similarity
    similarities = []
    for q1_embedding, q2_embedding in zip(question1_embeddings, question2_embeddings):
        similarity = cosine_similarity(
            q1_embedding.reshape(1, -1), q2_embedding.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)

    df["similarity"] = similarities
    return df

# Step 5: Query search function
def query_search(query, df, tokenizer, model):
    query_embedding = encode_text(query, tokenizer, model)

    similarities = []
    for question1 in df["question1"]:
        question_embedding = encode_text(str(question1), tokenizer, model)
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), question_embedding.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)

    df["query_similarity"] = similarities
    # Sort by similarity score
    df = df.sort_values(by="query_similarity", ascending=False)
    return df

# Step 6: Main function
def main():
    # Specify the path to the Quora dataset
    file_path = "data/questions.csv"  # Replace with your dataset file path

    # Load dataset
    df = prepare_quora_dataset(file_path)

    # Load BERT model and tokenizer
    tokenizer, model = load_bert_model()

    # Compute similarity scores for question pairs
    df_with_similarity = compute_similarity(df, tokenizer, model)

    # Perform a query search
    query = "How to learn programming?"  # Example query
    ranked_results = query_search(query, df_with_similarity, tokenizer, model)

    # Save the ranked results to a new CSV file
    output_path = "quora_query_search_results.csv"
    ranked_results.to_csv(output_path, index=False)
    print(f"Query search results saved to {output_path}")

if __name__ == "__main__":
    main()
