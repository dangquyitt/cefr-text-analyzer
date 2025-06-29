#!/usr/bin/env python3
"""
Simple training script for CEFR model
"""

from cefr_bert_classifier import CEFRTextAnalyzer
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
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
    dataset_file = 'dataset/dataset.csv'

    if not os.path.exists(dataset_file):
        print("❌ Training data not found!")
        print("Please ensure dataset/dataset.csv exists")
        return

    # Load and split data
    print("📊 Loading dataset...")
    df = pd.read_csv(dataset_file)
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())

    # Split data into train, validation and test sets (70-15-15 split)
    # First split: 70% train, 30% temp (which will be split into val and test)
    train_data, temp_data = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['label']  # Ensure balanced split across labels
    )

    # Second split: Split the 30% temp into 15% validation and 15% test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,  # 50% of 30% = 15% of total
        random_state=42,
        stratify=temp_data['label']
    )

    print(
        f"📊 Training samples: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(
        f"📊 Validation samples: {len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
    print(
        f"📊 Test samples: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")

    # Train the model using DataFrames directly
    print("🚀 Starting training...")
    analyzer.train_from_dataframes(
        train_data, val_data, epochs=3, best_model_path=model_path)

    print(f"✅ Best model saved as {model_path}")

    # Test the model on test set
    print("🧪 Testing model on test set...")
    test_accuracy = analyzer.evaluate_dataframe(test_data)
    print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")

    print("🎉 Training and testing completed!")


if __name__ == "__main__":
    main()
