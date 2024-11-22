import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification

class CybercrimeDataset(Dataset):
    def __init__(self, texts, categories=None, sub_categories=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.categories = categories
        self.sub_categories = sub_categories
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.categories is not None:
            item['category'] = torch.tensor(self.categories[idx], dtype=torch.long)
        if self.sub_categories is not None:
            item['sub_category'] = torch.tensor(self.sub_categories[idx], dtype=torch.long)
            
        return item

class HierarchicalBERTClassifier(torch.nn.Module):
    def __init__(self, model_name, num_categories, num_sub_categories):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.category_classifier = torch.nn.Linear(768, num_categories)
        self.sub_category_classifier = torch.nn.Linear(768, num_sub_categories)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.logits
        pooled_output = self.dropout(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        sub_category_logits = self.sub_category_classifier(pooled_output)
        return category_logits, sub_category_logits