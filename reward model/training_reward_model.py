import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class RankingDataset:
    def __init__(self, data_path, tokenizer, max_len=512, mode="train"):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.prepare_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        term = row['term']
        termVi_1 = row['termVi_1']
        termVi_2 = row['termVi_2']
        label = 0

        inputs_a = self.tokenizer(term, termVi_1, truncation=True, padding="max_length", max_length=self.max_len,
                                  return_tensors="pt")
        inputs_b = self.tokenizer(term, termVi_2, truncation=True, padding="max_length", max_length=self.max_len,
                                  return_tensors="pt")

        return {
            'input_ids_a': inputs_a['input_ids'].squeeze(),
            'attention_mask_a': inputs_a['attention_mask'].squeeze(),
            'input_ids_b': inputs_b['input_ids'].squeeze(),
            'attention_mask_b': inputs_b['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

    def prepare_data(self):
        rows = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for term in data:
            num_trans_term = len(data[0].keys()) - 1
            term['pairs'] = []
            for rank in range(1, num_trans_term):
                for lower_rank in range(rank + 1, num_trans_term + 1):
                    termVi_pair = (term[f"termVi_{rank}"], term[f"termVi_{lower_rank}"])
                    rows.append(
                        {
                            "term": term["term"],
                            "termVi_1": termVi_pair[0],
                            "termVi_2": termVi_pair[1],
                        }
                    )
        df = pd.DataFrame(rows)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=1508)
        if self.mode == "train":
            self.data = train_df
        else:
            self.data = val_df


def train_model(model, device, train_loader, val_loader, criterion, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            logits_a = model(input_ids_a, attention_mask_a).logits
            logits_b = model(input_ids_b, attention_mask_b).logits
            logits = logits_a - logits_b
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids_a = batch['input_ids_a'].to(device)
                attention_mask_a = batch['attention_mask_a'].to(device)
                input_ids_b = batch['input_ids_b'].to(device)
                attention_mask_b = batch['attention_mask_b'].to(device)
                labels = batch['label'].unsqueeze(1).to(device)

                logits_a = model(input_ids_a, attention_mask_a).logits
                logits_b = model(input_ids_b, attention_mask_b).logits
                logits = logits_a - logits_b
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")


def evaluate_model(model, device, tokenizer, term, termVis):
    model.eval()
    inputs = [tokenizer(term, termVi, truncation=True, padding="max_length", max_length=512, return_tensors="pt") for
              termVi in termVis]
    input_ids = torch.cat([i['input_ids'] for i in inputs]).to(device)
    attention_masks = torch.cat([i['attention_mask'] for i in inputs]).to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_masks)
        rewards = logits.squeeze(-1).cpu().numpy()

    return rewards


def main(strategy="normal", model_name="vinai/phobert-base-v2"):
    
    # init root model
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if strategy == "peft":
        from peft import get_peft_model, LoraConfig
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
    
        # Tạo DataLoader cho tập huấn luyện và tập kiểm tra
        data_path = "./DummyDatas.json"
        max_len = 258
        train_dataset = RankingDataset(data_path, tokenizer, max_len, mode="train")
        val_dataset = RankingDataset(data_path, tokenizer, max_len, mode="val")
    
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
    
        # Cấu hình LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "query",
                "key",
                "value"
            ]
        )
        # Áp dụng LoRA vào mô hình gốc
        model = get_peft_model(base_model, lora_config)
        model.to(device)
    
        # Khởi tạo criterion và optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
        train_model(model, device, train_loader, val_loader, criterion, optimizer, epochs=3)
    if strategy == "normal":
        pass
if __name__ == "__main__":
    main()
