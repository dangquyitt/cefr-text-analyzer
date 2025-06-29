# CEFR Text Analyzer with BERT

A BERT-based model for classifying English text proficiency levels according to CEFR standards (A1-C2). This project uses a fine-tuned BERT model to automatically assess the complexity level of English text.

## 🎯 Features

- **Automatic Classification**: Classify text into 6 CEFR levels (A1, A2, B1, B2, C1, C2)
- **BERT Model**: Uses BERT-base-uncased for high accuracy text classification
- **REST API**: FastAPI server with automatic model loading for easy integration
- **Device Support**: Automatic detection (CUDA/MPS/CPU) with mixed precision training
- **Simple Format**: Clean `text,label` dataset format for easy training

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
text,label
I am happy.,A1
I went to the store yesterday.,A2
I have been learning English for two years.,B1
The government has announced new policies.,B2
The comprehensive study reveals important insights.,C1
The epistemological considerations demand meticulous examination.,C2
```

- **text**: Text to be classified
- **label**: CEFR level (A1, A2, B1, B2, C1, C2)
- Simple two-column format for easy data preparation and training

## 🔧 API Usage

### Start the Server

```bash
python server.py
# Server runs on http://localhost:5050
# Model loads automatically at startup
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
│   ├── train.csv              # Training data (text,label format)
│   ├── validation.csv         # Validation data
│   └── test.csv               # Test data
├── cefr_bert_classifier.py    # Main BERT model implementation
├── server.py                  # FastAPI REST server
├── train.py                   # Training script
├── test_predictions.py        # Testing script
├── requirements.txt           # Python dependencies
├── cefr_bert_model.pth        # Trained model (after training)
└── README.md                  # This guide
```

## ⚙️ Model Parameters

- **Base Model**: BERT-base-uncased (109M parameters)
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 4 (training), 1 (inference)
- **Learning Rate**: 2e-5 with linear warmup
- **Training Epochs**: 3
- **Optimizer**: AdamW with weight decay (0.01)
- **Device**: Auto-detection (CUDA/MPS/CPU)
- **Mixed Precision**: Enabled for GPU acceleration
- **Model Size**: ~418MB when saved

## 📈 Model Output

The model provides:

- **Predicted Level**: A1-C2
- **Confidence Score**: 0.0-1.0
- **Batch Processing**: Multiple sentences at once
- **Detailed Metrics**: Accuracy, precision, recall per class

## 💡 Important Notes

1. **Dataset Format**: Uses simple `text,label` CSV format for easy data preparation
2. **Model Training**: Automatically saves only the best performing model (no duplicates)
3. **Server Startup**: Model loads at server startup, not per API request for better performance
4. **Device Detection**: Supports Apple Silicon (MPS), CUDA, and CPU with automatic selection
5. **Confidence Scores**: Low confidence (<0.5) may indicate text complexity between levels

## 🚀 Recent Updates

- ✅ Fixed duplicate model checkpoint creation during training
- ✅ Updated dataset format to simple `text,label` structure
- ✅ Improved server startup with automatic model loading
- ✅ Added PyTorch compatibility fixes for model loading
- ✅ Enhanced error handling and logging

## 🤝 References

- **BERT**: Hugging Face Transformers library
- **Framework**: PyTorch with Accelerate for GPU optimization
- **API**: FastAPI for high-performance REST API
- **CEFR Standards**: Common European Framework of Reference for Languages

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
rm cefr_bert_model.pth
python train.py
```

### Server Won't Start

```bash
# Check if model exists
ls -la cefr_bert_model.pth

# If not, train the model first
python train.py

# Then start server
python server.py
```

### PyTorch Version Issues

If you encounter model loading errors, ensure PyTorch compatibility:

```bash
pip install torch>=2.0.0
```

## 🚀 Getting Started

1. **Clone and setup**:

   ```bash
   cd cefr-text-analyzer
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Train the model**:

   ```bash
   python train.py
   # Wait for training to complete (~5-10 minutes)
   # Model will be saved as cefr_bert_model.pth
   ```

3. **Start the API server**:

   ```bash
   python server.py
   # Server starts on http://localhost:5050
   # Model loads automatically at startup
   ```

4. **Test the API**:

   ```bash
   # Simple test
   curl -X POST "http://localhost:5050/predict" \
        -H "Content-Type: application/json" \
        -d '{"sentences": ["Hello world!", "The comprehensive analysis reveals significant insights."]}'

   # Check server health
   curl http://localhost:5050/health

   # Get CEFR level information
   curl http://localhost:5050/levels
   ```

5. **Use in Python code**:

   ```python
   from cefr_bert_classifier import CEFRTextAnalyzer

   # Load trained model
   analyzer = CEFRTextAnalyzer()
   analyzer.load_model('cefr_bert_model.pth')

   # Make predictions
   level, confidence = analyzer.predict_text("This is a simple sentence.")
   print(f"CEFR Level: {level} (Confidence: {confidence:.3f})")
   ```

---

🎉 **Ready to use**: Your CEFR Text Analyzer API is now running!
