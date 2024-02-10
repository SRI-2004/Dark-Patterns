import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import joblib
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
class CustomDistilBERTModel(nn.Module):
    def __init__(self, distilbert, classification_head):
        super(CustomDistilBERTModel, self).__init__()
        self.distilbert = distilbert
        self.classification_head = classification_head

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state.mean(dim=1)  # Pooling over the sequence length
        predictions = self.classification_head(logits)
        return predictions




class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(in_dim, 2048)
        self.dense2 = nn.Linear(2048,4096)
        self.dense3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(4096,2048)
        self.dense5 = nn.Linear(2048, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.dense6 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dropout2(x)
        x = self.dense6(x)
        return x


# Data Input
selected_classification = "Pattern Category"
df = pd.read_csv('merged_file.csv')

# Data Preprocessing
df = df[pd.notnull(df["Pattern String"])]
col = ["Pattern String", selected_classification]
df = df[col]
df["category_id"], class_labels = df[selected_classification].factorize()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['Pattern String'], df['category_id'], train_size=0.3)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the training data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
train_labels = torch.tensor(list(y_train), dtype=torch.long)

# Tokenize and encode the testing data
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)
test_labels = torch.tensor(list(y_test), dtype=torch.long)

# Create dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

# Load pre-trained DistilBERT model
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Create the custom model
num_labels = len(class_labels)
classification_head = ClassificationHead(distilbert.config.hidden_size, num_labels)
model = CustomDistilBERTModel(distilbert, classification_head)

# Fine-tune the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_losses = []
train_accuracies = []

# Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0.0
    correct_predictions = 0

    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask=attention_mask)

        # Calculate loss
        loss = nn.CrossEntropyLoss()(predictions, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optim.step()

        # Calculate accuracy
        correct_predictions += torch.sum(torch.argmax(predictions, dim=1) == labels).item()

    # Calculate average loss and accuracy for the epoch
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / len(train_loader.dataset)

    # Save metrics
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f'Epoch {epoch + 1}/{10} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Plot training metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluation
model.eval()
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask=attention_mask)

        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Convert to arrays before calculating accuracy
y_test_array = test_labels.numpy()
all_preds_array = np.array(all_preds)

# Print accuracy
accuracy = accuracy_score(y_test_array, all_preds_array)
print("Accuracy:", accuracy)

# Save the fine-tuned model
conf_mat = confusion_matrix(y_test_array, all_preds_array)
plt.figure(figsize=(8, 8))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
# print("Classification Report:\n", classification_report(y_test_array, all_preds_array, target_names=class_labels))

# Calculate F1 score
f1 = f1_score(y_test_array, all_preds_array, average='weighted')
print("F1 Score:", f1)

# Save the fine-tuned model
joblib.dump(model, 'models/category_classifier_bert.joblib')
joblib.dump(tokenizer, 'models/category_tokenizer_bert.joblib')
