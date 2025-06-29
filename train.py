#!/usr/bin/env python3
"""
Simple training script for CEFR model
"""

from cefr_bert_classifier import CEFRTextAnalyzer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("🔄 Training CEFR BERT Model...")

    # Check if model already exists
    model_path = 'cefr_bert_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return

    # Initialize analyzer
    analyzer = CEFRTextAnalyzer(
        model_name='bert-base-uncased',
        max_length=128,
        batch_size=4,
        learning_rate=2e-5
    )

    # Check if training data exists
    train_file = 'dataset/train.csv'
    val_file = 'dataset/validation.csv'

    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        print("❌ Training data not found!")
        print("Please ensure dataset/train.csv and dataset/validation.csv exist")
        return

    # Train the model
    print("🚀 Starting training...")
    analyzer.train(train_file, val_file, epochs=3, best_model_path=model_path)

    print(f"✅ Best model saved as {model_path}")
    print("🎉 Training completed!")


if __name__ == "__main__":
    main()
