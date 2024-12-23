from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

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
        
    def forward(self, query_text, doc_text):
        # Tokenize and encode the query and document
        query_inputs = self.tokenizer(query_text, padding=True, truncation=True, 
                                    max_length=128, return_tensors="pt")
        doc_inputs = self.tokenizer(doc_text, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt")
        
        # Get BERT embeddings
        query_outputs = self.bert(**query_inputs)[1]  # Use [CLS] token embedding
        doc_outputs = self.bert(**doc_inputs)[1]
        
        # Concatenate query and document embeddings
        combined = torch.cat([query_outputs, doc_outputs], dim=1)
        
        # Pass through neural network layers
        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)
