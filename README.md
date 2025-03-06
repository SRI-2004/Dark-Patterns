---

## **Project Recognition**  
- 🏆 **Finalist at Dark Patterns Buster Hackathon (IIT BHU, Ministry of Consumer Affairs, GOI)**  
- Built in collaboration with:  
  - **[Vishruth R](https://github.com/vishruth-r)**  
  - **[Anirudh Arcot](https://github.com/cr4z4x)**
  - **[Devangana Ghosh](https://github.com/Devangana-Ghosh)**
  - **[Aahanaa Sharrma](https://github.com/aahanaasharrma)**    

---

# **Dark Patterns Detector: AI-Powered Identification of Deceptive UI/UX Practices**  

🏆 **Finalist at Dark Patterns Buster Hackathon (IIT BHU, Ministry of Consumer Affairs, GOI)**  

## **Overview**  
Dark patterns are **deceptive UI/UX practices** that manipulate users into making unintended choices. This project leverages **NLP-based deep learning models** to **detect and classify dark patterns** in textual descriptions and website interactions.  

We built a **DistilBERT-based classification model** capable of identifying **various types of dark patterns** with high accuracy. The model was integrated into a **Django web app and a browser extension**, enabling real-time detection of deceptive design elements.  

---

## **Key Features**  
- **DistilBERT-Based Dark Pattern Classifier**  
  - Custom **Classification Head** with **deep feedforward layers** for enhanced feature learning.  
  - Trained on a **dark pattern dataset** to distinguish deceptive UI/UX elements.  
  - Implements **dropout regularization** to improve generalization and robustness.  

- **Real-Time Detection & Analysis**  
  - **Django Web App** for **classifying website content** based on dark pattern severity.  
  - **Browser Extension** (Built with JavaScript & Flask API) to analyze **live web pages**.  

- **Category-Based Classification**  
  - Identifies dark patterns across categories like:  
    - **Forced Continuity** (e.g., hidden subscriptions)  
    - **Nagging** (e.g., repeated pop-ups)  
    - **Sneaking** (e.g., hidden costs)  
    - **Obstruction** (e.g., deliberately difficult cancellation flows)  
    - **Interface Interference** (e.g., misleading button placements)  

---

## **Tech Stack**  

- **Machine Learning & NLP**  
  - **DistilBERT** for text feature extraction  
  - **Custom Feedforward Classification Head**  
  - **PyTorch, Hugging Face Transformers**  

- **Web & API Backend**  
  - **Django, Flask** (for web classification API)  
  - **PostgreSQL** (database for tracking flagged patterns)  

- **Browser Extension**  
  - **JavaScript, Chrome API** (frontend)  
  - **Flask API** (backend for text processing & classification)  

---

## **Model Architecture**  

### **DistilBERT-Based Classifier**  
The classification model consists of:  
- **Pretrained DistilBERT Encoder** (for extracting semantic features from text).  
- **Custom Multi-Layer Feedforward Network**:  
  - **2048 → 4096 → 2048 → 512 → Output Classes**  
  - **Dropout Layers** for generalization.  
  - **ReLU activations** for non-linearity.  

```python
class CustomDistilBERTModel(nn.Module):
    def __init__(self, distilbert, classification_head):
        super(CustomDistilBERTModel, self).__init__()
        self.distilbert = distilbert
        self.classification_head = classification_head

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence
        predictions = self.classification_head(logits)
        return predictions
```

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/dark-patterns-detector.git
cd dark-patterns-detector
```

### **2. Install Dependencies**  
```bash
pip install torch transformers django flask requests beautifulsoup4
```

### **3. Run the Web Classifier**  
```bash
python manage.py runserver
```

### **4. Browser Extension Setup**  
1. Load the **Chrome extension** from the `browser_extension/` directory.  
2. Start the **Flask API** for classification.  

---

