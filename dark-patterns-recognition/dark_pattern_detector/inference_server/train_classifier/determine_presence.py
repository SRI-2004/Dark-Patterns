import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn import metrics
from joblib import dump
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn

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



# Read data
df1 = pd.read_csv('normie.csv')
df2 = pd.read_csv('dark_patterns.csv')

df1 = df1[pd.notnull(df1["Pattern String"])]
df1 = df1[df1["classification"] == 0]
df1["classification"] = "Not Dark"
df1.drop_duplicates(subset="Pattern String")

df2 = df2[pd.notnull(df2["Pattern String"])]
df2["classification"] = "Dark"
col = ["Pattern String", "classification"]
df2 = df2[col]

df = pd.concat([df1, df2])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Pattern String'], df["classification"], train_size=0.3)

# Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbertmodel = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define custom classification head

# Custom Classification Head

classification_head = ClassificationHead(distilbertmodel.config.hidden_size, 2)
# Combine DistilBERT and the custom classification head
model = CustomDistilBERTModel(distilbertmodel, classification_head)  # Assuming binary classification

# Tokenize and encode the training data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
train_labels = torch.tensor((y_train.to_numpy() == "Dark").astype(int), dtype=torch.long)  # Convert labels to 0 or 1

# Tokenize and encode the testing data
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)
test_labels = torch.tensor((y_test.to_numpy() == "Dark").astype(int), dtype=torch.long)  # Convert labels to 0 or 1

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

# Create DataLoader
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# Fine-tune the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_losses = []
train_accuracies = []
train_f1_scores = []

for epoch in range(3):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    all_true_labels = []
    all_pred_labels = []

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(predictions, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(predictions, 1)
        correct_predictions += (predicted_labels == labels).sum().item()

        all_true_labels.extend(labels.cpu().numpy())
        all_pred_labels.extend(predicted_labels.cpu().numpy())

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / len(train_loader.dataset)
    epoch_f1_score = f1_score(all_true_labels, all_pred_labels, average='binary')

    # Save metrics
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    train_f1_scores.append(epoch_f1_score)

    print(f'Epoch {epoch + 1}/{10} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1 Score: {epoch_f1_score:.4f}')

# Evaluate the model
model.eval()
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
all_preds = []
all_true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(predictions, 1)

        all_preds.extend(predicted_labels.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# Calculate metrics on test set
accuracy = metrics.accuracy_score(all_true_labels, all_preds)
f1 = f1_score(all_true_labels, all_preds, average='binary')
fpr, tpr, thresholds = roc_curve(all_true_labels, all_preds)
roc_auc = auc(fpr, tpr)

print("Test Accuracy:", accuracy)
print("Test F1 Score:", f1)
print("Test AUC:", roc_auc)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

joblib.dump(model, 'models/presence_classifier_distilbert.joblib')
joblib.dump(tokenizer, 'models/presence_tokenizer_distilbert.joblib')

