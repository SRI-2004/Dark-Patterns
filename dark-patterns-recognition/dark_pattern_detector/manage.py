#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
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

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dark_pattern_detector.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
