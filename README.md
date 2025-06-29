# CEFR Text Analyzer with BERT

A BERT-based model for classifying English text proficiency levels according to CEFR standards (A1-C2), built with reference to the CEFR-SP dataset.

## 🎯 Features

- **Automatic Classification**: Classify text into 6 CEFR levels (A1, A2, B1, B2, C1, C2)
- **BERT Model**: Uses BERT-base-uncased for high accuracy
- **REST API**: FastAPI server for easy integration
- **GPU/CPU Support**: Automatic device detection with mixed precision training

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
cd /path/to/cefr-text-analyzer
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train.py
```

### 4. Start API Server

```bash
python server.py
```

## 📊 Data Structure

Training data in CSV format:

```csv
text,Annotator I,Annotator II
I bought both of them.,1,1
The weather is nice today.,2,2
She has been studying for years.,3,3
```

- **text**: Text to be classified
- **Annotator I/II**: Labels from 2 annotators (1=A1, 2=A2, ..., 6=C2)
- Model uses the average of 2 annotators as ground truth

## 🔧 API Usage

### Start the Server

```bash
python server.py
# Server runs on http://localhost:5050
```

### Make Predictions

```bash
curl -X POST "http://localhost:5050/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sentences": [
         "This repository contains a model trained to predict Common European Framework of Reference levels."
       ]
     }'
```

### Response Format

```json
[
  {
    "cefr": "B2",
    "sentence": "This repository contains a model trained to predict Common European Framework of Reference levels.",
    "confidence": 0.8756
  }
]
```

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict CEFR levels
- `GET /levels` - CEFR level descriptions

### Direct Usage in Code

```python
from cefr_bert_classifier import CEFRTextAnalyzer

# Initialize
analyzer = CEFRTextAnalyzer()

# Load trained model
analyzer.load_model('cefr_bert_model.pth')

# Predict
level, confidence = analyzer.predict_text("The implementation requires careful consideration.")
print(f"CEFR Level: {level} (Confidence: {confidence:.3f})")
```

## 📚 CEFR Levels Description

| Level  | Name               | Description                                   |
| ------ | ------------------ | --------------------------------------------- |
| **A1** | Beginner           | Basic everyday expressions and simple phrases |
| **A2** | Elementary         | Simple sentences on familiar topics           |
| **B1** | Intermediate       | Clear communication on familiar matters       |
| **B2** | Upper Intermediate | Complex texts and abstract topics             |
| **C1** | Advanced           | Flexible and effective language use           |
| **C2** | Proficient         | Very high level with nuanced expression       |

## 📁 Project Structure

```
cefr-text-analyzer/
├── .venv/                      # Python virtual environment
├── dataset/                    # Training/test data
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── CEFR-SP/                    # Reference from CEFR-SP
├── cefr_bert_classifier.py     # Main BERT model
├── server.py                   # FastAPI REST server
├── train.py                    # Training script
├── requirements.txt            # Dependencies
└── README.md                   # This guide
```

## ⚙️ Model Parameters

- **Base Model**: BERT-base-uncased
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 4-8 (depending on memory)
- **Learning Rate**: 2e-5
- **Training Epochs**: 3-5
- **Optimizer**: AdamW with weight decay
- **Device**: Auto-detection (CUDA/CPU)
- **Mixed Precision**: Enabled for GPU training

## 📈 Model Output

The model provides:

- **Predicted Level**: A1-C2
- **Confidence Score**: 0.0-1.0
- **Batch Processing**: Multiple sentences at once
- **Detailed Metrics**: Accuracy, precision, recall per class

## 💡 Important Notes

1. **Small Dataset**: Currently trained on a small dataset, accuracy may be limited
2. **Improvements**: For better results, consider:
   - Increasing training data size
   - Fine-tuning hyperparameters
   - Using domain-specific pre-trained models
3. **Confidence**: Low confidence scores (<0.4) may indicate need for more context

## 🤝 References

- **CEFR-SP Dataset**: Structure and ideas from CEFR-SP
- **BERT**: Hugging Face Transformers
- **Framework**: PyTorch with Accelerate

## 🛠️ Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA/Memory Issues

```python
# Reduce batch_size in code
analyzer = CEFRTextAnalyzer(batch_size=2)
```

### Model Loading Issues

```bash
# Remove old model and retrain
rm cefr_bert_model.pth best_cefr_model.pth
python train.py
```

## 🚀 Getting Started

1. **Clone and setup**:

   ```bash
   git clone <your-repo>
   cd cefr-text-analyzer
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train model**:

   ```bash
   python train.py
   ```

3. **Start API server**:

   ```bash
   python server.py
   ```

4. **Test API**:
   ```bash
   curl -X POST "http://localhost:5050/predict" \
        -H "Content-Type: application/json" \
        -d '{"sentences": ["Hello world!"]}'
   ```

---

🎉 **Ready to use**: Your CEFR Text Analyzer API is now running!
