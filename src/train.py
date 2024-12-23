""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class RankingDataset(Dataset):
    def __init__(self, queries, docs, labels):
        self.queries = queries
        self.docs = docs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.queries[idx], self.docs[idx], self.labels[idx]

class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(RankingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1) 
        )

    def forward(self, query, doc):
        x = torch.cat([query, doc], dim=1)
        return self.fc(x)
        
def train_model(data_path, model_path, epochs=1500, batch_size=64, patience=5):
    # Load preprocessed data
    data = np.load(data_path)
    queries, docs, labels = data['query_vectors'], data['doc_vectors'], data['labels']

    # Prepare dataset and dataloader
    dataset = RankingDataset(queries, docs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model, loss, and optimizer
    input_size = queries.shape[1] * 2
    model = RankingModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 

    best_loss = None
    wait = 0

    # Training loop
    for epoch in range(epochs):
        for query, doc, label in dataloader:
            query, doc, label = query.float(), doc.float(), label.float()
            outputs = model(query, doc).squeeze()
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            if best_loss is None or loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
                torch.save(model.state_dict(), model_path)  # Save best model
            else:
                wait += 1
                if wait == patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model("data/preprocessed_data.npz", "models/ranking_model.pth")
 """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

class RankingDataset(Dataset):
    def __init__(self, queries, docs, labels, tokenizer, max_length=128):
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Pre-tokenize the data
        query_inputs = self.tokenizer(
            self.queries[idx], 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        doc_inputs = self.tokenizer(
            self.docs[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'query_inputs': {k: v.squeeze(0) for k, v in query_inputs.items()},
            'doc_inputs': {k: v.squeeze(0) for k, v in doc_inputs.items()},
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class BertRankingModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BertRankingModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Neural network layers after BERT
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 2, 256)  # 768 is BERT's hidden size
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, query_inputs, doc_inputs):
        query_outputs = self.bert(**query_inputs)[1]
        doc_outputs = self.bert(**doc_inputs)[1]
        
        # Concatenate embeddings
        combined = torch.cat([query_outputs, doc_outputs], dim=1)
        
        # Pass through neural network layers
        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)

def train_model(data_path, model_path, epochs=6, batch_size=32, patience=3):
    # Load preprocessed data
    data = np.load(data_path, allow_pickle=True)
    queries, docs, labels = data['query_texts'], data['doc_texts'], data['labels']
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare dataset and dataloader
    dataset = RankingDataset(queries, docs, labels, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BertRankingModel().to(device)
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    criterion = nn.MSELoss()
    best_loss = float('inf')
    wait = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            query_inputs = {k: v.to(device) for k, v in batch['query_inputs'].items()}
            doc_inputs = {k: v.to(device) for k, v in batch['doc_inputs'].items()}
            labels = batch['labels'].to(device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(query_inputs, doc_inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model("data/preprocessed_data.npz", "models/bert_ranking_model.pth")