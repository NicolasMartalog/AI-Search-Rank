import torch
import numpy as np
from train import BertRankingModel
from tqdm import tqdm
import os
from transformers import BertTokenizer, BertModel
import pandas as pd

def rank_all_queries(data_path, model_path):
    # Load preprocessed data
    data = np.load(data_path)
    queries, docs, labels = data['query_vectors'], data['doc_vectors'], data['labels']

    # Load model
    input_size = queries.shape[1] * 2
    model = RankingModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Rank documents
    rankings = []
    with torch.no_grad():
        for i in range(len(queries)):
            query = torch.tensor(queries[i]).float().unsqueeze(0)
            doc = torch.tensor(docs[i]).float().unsqueeze(0)
            score = model(query, doc).item()
            rankings.append((i, score))

    # Sort rankings by score
    rankings.sort(key=lambda x: x[1], reverse=True)
    print("Rankings (Index, Score):", rankings)
    return rankings 

def rank_query_string(query_string, data_path, model_path, original_path, n, tokenizer_path='bert-base-uncased'):
    # Load preprocessed data
    data = np.load(data_path)
    docs = data['doc_vectors']

    # Load tokenizer and BERT model
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    bert_model = BertModel.from_pretrained(tokenizer_path)
    bert_model.eval()

    # Encode the query string using BERT
    inputs = tokenizer(query_string, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    # Resize query_vector to match training dimensions
    input_size = docs.shape[1]  # Get the document vector size from training data
    query_vector = query_vector[:input_size]  # Truncate to match document vector size
    if query_vector.shape[0] < input_size:    # Pad if necessary
        query_vector = np.pad(query_vector, (0, input_size - query_vector.shape[0]))

    # Load ranking model with correct input size
    model = RankingModel(input_size * 2)  # Multiply by 2 because we concatenate query and doc vectors
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Rank the single query against all documents
    rankings = []
    with torch.no_grad():
        query_tensor = torch.tensor(query_vector).float().unsqueeze(0)
        for i, doc_vector in enumerate(docs):
            doc_tensor = torch.tensor(doc_vector).float().unsqueeze(0)
            score = model(query_tensor, doc_tensor).item()
            rankings.append((i, score))

    # Sort rankings by score
    rankings.sort(key=lambda x: x[1], reverse=True)
    t_n_ranking = list(rankings[:n])
    res = []

    df = pd.read_csv(original_path)

    for item in t_n_ranking: 
        print(item[0])
        res.append((df.iloc[item[0]]['document'], item[1]))

    return res 

def rank_query_string_new(query_string, model_path, original_path, n=5, device='cpu'):
    # Load the trained model
    model = BertRankingModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load original documents
    df = pd.read_csv(original_path)
    documents = df['document'].values
    
    # Rank documents
    rankings = []
    with torch.no_grad():
        for i, doc in enumerate(documents):
            score = model(query_string, doc).item()
            rankings.append((i, score, doc))
    
    # Sort by score and get top n
    rankings.sort(key=lambda x: x[1], reverse=True)
    top_n = rankings[:n]
    
    print(f"\nTop {n} results for query: '{query_string}'")
    print("-" * 50)
    for i, (idx, score, doc) in enumerate(top_n, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"Document: {doc}\n")
    
    return top_n 

def find_similar_questions(query_question, model_path, questions_path, n=5, device='cpu', max_questions=1000):
    # Load the trained model
    model = BertRankingModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load questions and sample a subset for faster inference
    df = pd.read_csv(questions_path)
    all_questions = pd.concat([df['question1'], df['question2']]).unique()
    
    if len(all_questions) > max_questions:
        np.random.seed(42)
        all_questions = np.random.choice(all_questions, max_questions, replace=False)
    
    print(f"Comparing against {len(all_questions)} questions...")
    
    # Batch processing for faster inference
    batch_size = 32
    rankings = []
    
    # Pre-tokenize query once
    query_encoding = model.tokenizer(
        query_question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    query_inputs = {k: v.to(device) for k, v in query_encoding.items()}
    
    # Get query embedding once
    with torch.no_grad():
        query_embedding = model.bert(**query_inputs)[1]
    
    # Process questions in batches
    for i in tqdm(range(0, len(all_questions), batch_size)):
        batch_questions = all_questions[i:i + batch_size]
        
        # Tokenize batch
        batch_encoding = model.tokenizer(
            list(batch_questions),
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        batch_inputs = {k: v.to(device) for k, v in batch_encoding.items()}
        
        with torch.no_grad():
            # Get document embeddings
            doc_embeddings = model.bert(**batch_inputs)[1]
            
            # Expand query embedding to match batch size
            query_expanded = query_embedding.expand(doc_embeddings.shape[0], -1)
            
            # Concatenate and pass through the rest of the model
            combined = torch.cat([query_expanded, doc_embeddings], dim=1)
            x = model.dropout(combined)
            x = model.relu(model.fc1(x))
            x = model.relu(model.fc2(x))
            scores = model.fc3(x).squeeze(-1)
            
            # Add to rankings
            for j, score in enumerate(scores):
                if batch_questions[j] != query_question:
                    rankings.append((i+j, score.item(), batch_questions[j]))
    
    # Sort by score and get top n
    rankings.sort(key=lambda x: x[1], reverse=True)
    top_n = rankings[:n]
    
    print(f"\nTop {n} similar questions for: '{query_question}'")
    print("-" * 50)
    for i, (idx, score, question) in enumerate(top_n, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"Question: {question}\n")
    
    return top_n

if __name__ == "__main__":
    query = "good videos for Java or Python?"
    results = find_similar_questions(
        query_question=query,
        model_path="models/bert_ranking_model.pth",
        questions_path="data/questions.csv",
        n=5, 
        max_questions=1000
    )
