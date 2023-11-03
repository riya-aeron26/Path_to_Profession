import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Define your data (X: input text, y: labels)
X = ['Your text data here']
y = [label]

# Tokenize and encode the data
tokenized = [tokenizer.encode(text, add_special_tokens=True) for text in X]
padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tokenized], batch_first=True)
attention_mask = (padded != 0).type(torch.FloatTensor)

# Create a TensorDataset and split it into train and validation sets
dataset = TensorDataset(padded, attention_mask, torch.tensor(y, dtype=torch.long))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader objects for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Validation Epoch {epoch + 1}'):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    print(f'Epoch {epoch + 1}, Training Loss: {train_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}')

# Save the trained model
model.save_pretrained('your_model_directory')

# You can later load the model for inference
# model = BertForSequenceClassification.from_pretrained('your_model_directory')
