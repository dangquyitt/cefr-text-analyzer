# CEFR Text Analyzer

A machine learning system for classifying English text according to CEFR proficiency levels (A1-C2) using BERT-based classification.

## Features

- **Pure Classification Approach**: No regression or contrastive learning
- **Class-Weighted Loss**: Handles imbalanced datasets effectively
- **Mean Pooling**: Better sentence representation than [CLS] token
- **F1 Score Evaluation**: More suitable for imbalanced classification
- **REST API**: Easy integration with web applications

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load data from `dataset/dataset.csv`
- Train the BERT-based classifier
- Save the best model as `cefr_bert_model.pth`
- Show evaluation metrics and sample predictions

### 3. Start the API Server

```bash
python server.py
```

The server will run on `http://localhost:5050`

## API Usage

### Health Check
```bash
curl http://localhost:5050/health
```

### Predict CEFR Levels
```bash
curl -X POST "http://localhost:5050/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentences": ["I like cats.", "The implementation requires careful consideration."]}'
```

### Get CEFR Level Information
```bash
curl http://localhost:5050/levels
```

## CEFR Levels

- **A1 (Beginner)**: Basic everyday expressions and very simple phrases
- **A2 (Elementary)**: Simple sentences on familiar topics and routine matters
- **B1 (Intermediate)**: Clear communication on familiar matters and personal interests
- **B2 (Upper Intermediate)**: Complex texts and abstract topics with good fluency
- **C1 (Advanced)**: Flexible and effective language use for social, academic and professional purposes
- **C2 (Proficient)**: Very high level with precise, nuanced expression and full command of language

## Key Improvements

✓ **Pure classification** (no regression)  
✓ **Class-weighted loss** for imbalanced data  
✓ **Mean pooling** for better sentence representation  
✓ **F1 score evaluation** (better for imbalanced classes)  
✓ **Gradient clipping** for training stability  
✓ **Based on CEFR-SP research methodology**

## Files

- `model.py`: Main CEFR classifier implementation
- `train.py`: Training script
- `server.py`: FastAPI REST API server
- `dataset/dataset.csv`: Training data (required)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- scikit-learn
- pandas
- numpy
