""" import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_data(input_path, output_path):
    # Load the dataset
    data = pd.read_csv(input_path)

    # Generate relevance scores from clicks and impressions
    data['relevance'] = data['clicks'] / (data['impressions'] + 1)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    query_vectors = vectorizer.fit_transform(data['query'])
    doc_vectors = vectorizer.transform(data['document'])

    # Save preprocessed data
    np.savez(output_path, 
             query_vectors=query_vectors.toarray(),
             doc_vectors=doc_vectors.toarray(),
             labels=data['relevance'].values)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/query-document.csv", "data/preprocessed_data.npz") """
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_path, max_samples=50000):
    # Load the dataset
    data = pd.read_csv(input_path)
    
    # Balance the dataset
    duplicate_pairs = data[data['is_duplicate'] == 1]
    non_duplicate_pairs = data[data['is_duplicate'] == 0]
    
    # Sample equal numbers of duplicate and non-duplicate pairs
    samples_per_class = min(len(duplicate_pairs), len(non_duplicate_pairs), max_samples // 2)
    duplicate_pairs = duplicate_pairs.sample(n=samples_per_class, random_state=42)
    non_duplicate_pairs = non_duplicate_pairs.sample(n=samples_per_class, random_state=42)
    
    # Combine the balanced dataset
    balanced_data = pd.concat([duplicate_pairs, non_duplicate_pairs])
    balanced_data = balanced_data.sample(frac=1, random_state=42)  # Shuffle
    
    # Extract pairs and labels
    queries = balanced_data['question1'].values
    docs = balanced_data['question2'].values
    labels = balanced_data['is_duplicate'].values
    
    # Save preprocessed data
    np.savez(output_path,
             query_texts=queries,
             doc_texts=docs,
             labels=labels)
    
    print(f"Preprocessed data saved to {output_path}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of duplicate pairs: {np.sum(labels)}")
    print(f"Number of non-duplicate pairs: {len(labels) - np.sum(labels)}")
    print(f"Percentage of duplicates: {(np.sum(labels)/len(labels))*100:.2f}%")

if __name__ == "__main__":
    preprocess_data("data/questions.csv", "data/preprocessed_data.npz", max_samples=50000)