
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from joblib import load

from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier, LogisticRegression
device = "cuda"
# Read data
df1 = pd.read_csv('normie.csv')
df2 = pd.read_csv('dark_patterns_new.csv')

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
    df['Pattern String'], df["classification"], train_size=.25, random_state=42)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# Tokenizer and Model
tokenizer = load("../api/presence_tokenizer_distilbert.joblib")
# Load your pre-trained DistilBERT model
distilbert_model = load('../api/presence_classifier_distilbert.joblib')

# Tokenize and encode the testing data for DistilBERT
test_encodings_distilbert = tokenizer(list(X_test), truncation=True, padding=True)
test_labels_distilbert = torch.tensor((y_test.to_numpy() == "Dark").astype(int), dtype=torch.long)  # Convert labels to 0 or 1

# Create dataset class for DistilBERT
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

# Create DataLoader for DistilBERT
test_dataset_distilbert = TextDataset(test_encodings_distilbert, test_labels_distilbert)
test_loader_distilbert = DataLoader(test_dataset_distilbert, batch_size=8, shuffle=False)

# Evaluate the DistilBERT model
distilbert_model.eval()
all_preds_distilbert = []
all_true_labels_distilbert = []

with torch.no_grad():
    for batch in tqdm(test_loader_distilbert):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = distilbert_model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(predictions, 1)

        all_preds_distilbert.extend(predicted_labels.cpu().numpy())
        all_true_labels_distilbert.extend(labels.cpu().numpy())

# Calculate accuracy on the test set for DistilBERT
accuracy_distilbert = metrics.accuracy_score(all_true_labels_distilbert, all_preds_distilbert)

# Print accuracy for DistilBERT
print("DistilBERT Accuracy:", accuracy_distilbert)



# Model creation

classifiers = []
accs = []
cms = []

classifiers.append(BernoulliNB())
classifiers.append(MultinomialNB())
classifiers.append(RandomForestClassifier())
classifiers.append(svm.SVC())
classifiers.append(tree.DecisionTreeClassifier())
classifiers.append(SGDClassifier())
classifiers.append(LogisticRegression())

for clf in classifiers:
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(count_vect.transform(X_test))
    accs.append(metrics.accuracy_score(y_test, y_pred))
    cms.append(metrics.confusion_matrix(y_test, y_pred))

for i in range(len(classifiers)):
    print(f"{classifiers[i]} accuracy: {accs[i]}")
    print(f"Confusion Matris: {cms[i]}")