from joblib import load
import shap
import pandas as pd
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
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
        self.dropout1 = nn.Dropout(0.8)
        self.dense1 = nn.Linear(in_dim, 2048)
        self.dense2 = nn.Linear(2048, 4096)
        self.dense3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.8)
        self.dense4 = nn.Linear(4096, 2048)
        self.dense5 = nn.Linear(2048, 512)
        self.dropout3 = nn.Dropout(0.8)
        self.dense6 = nn.Linear(512, out_dim)
        self.dense7 = nn.Linear(in_dim,out_dim)
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


# Load your trained model
import torch
# Create an instance of your model
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
classification_head = ClassificationHead(distilbert.config.hidden_size, 7)
model = CustomDistilBERTModel(distilbert,classification_head)

# Load the saved model state dictionary
model.load_state_dict(torch.load('category_model.pth'))

# Set the model in evaluation mode
model.eval()

# Load your dataset
dataset = pd.read_csv("merged_file.csv")
X = dataset['Pattern String']
y = dataset['Pattern Category']
# Initialize the SHAP explainer with your model and data
explainer = shap.Explainer(model,dataset)

# Calculate SHAP values for a specific instance
# Load the DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize your text data
input_ids = tokenizer(X.iloc[0], return_tensors='pt')['input_ids']
attention_mask = tokenizer(X.iloc[0], return_tensors='pt')['attention_mask']

# Pass the numerical data to the shap.Explainer
shap_values = explainer(input_ids, attention_mask)

# Visualize the SHAP values


if __name__ == '__main__':
    shap.plots.waterfall(shap_values)