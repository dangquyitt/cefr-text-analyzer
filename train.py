#!/usr/bin/env python3
"""
Simple training script for CEFR model
"""

from model import CEFRTextAnalyzer
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib/Seaborn not available. Skipping visualization plots.")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_detailed_reports(analyzer, predictions, true_labels, test_texts):
    """Generate comprehensive evaluation reports for classification model"""

    # CEFR levels for visualization
    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*80)

    # 1. Classification Report with per-class metrics
    print("\n📋 CLASSIFICATION REPORT (Per-Class Metrics):")
    print("-" * 60)
    report = classification_report(
        true_labels, predictions,
        labels=list(range(6)),
        target_names=cefr_levels,
        digits=4,
        zero_division=0
    )
    print(report)

    # 2. Confusion Matrix
    print("\n🔄 CONFUSION MATRIX:")
    print("-" * 40)
    cm = confusion_matrix(true_labels, predictions, labels=list(range(6)))

    # Print confusion matrix with labels
    print("Predicted →")
    print("     ", " ".join(f"{level:>6}" for level in cefr_levels))
    print("True ↓")
    for i, level in enumerate(cefr_levels):
        row = " ".join(f"{cm[i][j]:>6}" for j in range(6))
        print(f"{level:>4}: {row}")

    # 3. Per-class accuracy analysis
    print("\n🎯 PER-CLASS PERFORMANCE:")
    print("-" * 50)
    for i, level in enumerate(cefr_levels):
        class_mask = np.array(true_labels) == i
        if np.sum(class_mask) > 0:  # Only if class exists in test set
            class_predictions = np.array(predictions)[class_mask]
            class_accuracy = np.mean(class_predictions == i)
            class_precision = precision_score(true_labels, predictions, labels=[
                                              i], average=None, zero_division=0)
            class_recall = recall_score(true_labels, predictions, labels=[
                                        i], average=None, zero_division=0)

            if len(class_precision) > 0:
                print(
                    f"{level}: Accuracy={class_accuracy:.3f}, Precision={class_precision[0]:.3f}, Recall={class_recall[0]:.3f}")

    # 4. Confidence Analysis
    print("\n🔍 CONFIDENCE ANALYSIS:")
    print("-" * 30)
    analyze_confidence_distribution(
        analyzer, test_texts, true_labels, predictions)

    # 5. Error Analysis - Show misclassified samples
    print("\n❌ ERROR ANALYSIS (Sample Misclassifications):")
    print("-" * 55)
    analyze_errors(test_texts, true_labels, predictions,
                   cefr_levels, max_samples=10)

    # 6. Class Distribution Analysis
    print("\n📈 CLASS DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    analyze_class_distribution(true_labels, predictions, cefr_levels)

    # 7. Generate visualization if possible
    if PLOTTING_AVAILABLE:
        print("\n🎨 Generating visualization plots...")
        plot_confusion_matrix(cm, cefr_levels)
        plot_class_distribution(true_labels, predictions, cefr_levels)

    print("\n" + "="*80)
    print("✅ EVALUATION REPORT COMPLETED")
    print("="*80)


def analyze_confidence_distribution(analyzer, test_texts, true_labels, predictions):
    """Analyze confidence distribution of predictions"""

    # Get confidence scores for all predictions
    confidences = []
    correct_confidences = []
    incorrect_confidences = []

    print("Computing confidence scores...")
    for i, text in enumerate(test_texts):
        _, confidence = analyzer.predict_text(text)
        confidences.append(confidence)

        if predictions[i] == true_labels[i]:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

    # Statistics
    print(
        f"Overall confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
    print(
        f"Correct predictions - Mean: {np.mean(correct_confidences):.3f}, Std: {np.std(correct_confidences):.3f}")
    print(
        f"Incorrect predictions - Mean: {np.mean(incorrect_confidences):.3f}, Std: {np.std(incorrect_confidences):.3f}")

    # Confidence ranges
    high_conf = np.sum(np.array(confidences) > 0.8)
    medium_conf = np.sum((np.array(confidences) > 0.6) &
                         (np.array(confidences) <= 0.8))
    low_conf = np.sum(np.array(confidences) <= 0.6)

    print(
        f"High confidence (>0.8): {high_conf} samples ({high_conf/len(confidences)*100:.1f}%)")
    print(
        f"Medium confidence (0.6-0.8): {medium_conf} samples ({medium_conf/len(confidences)*100:.1f}%)")
    print(
        f"Low confidence (≤0.6): {low_conf} samples ({low_conf/len(confidences)*100:.1f}%)")


def analyze_errors(test_texts, true_labels, predictions, cefr_levels, max_samples=10):
    """Analyze misclassified samples"""

    errors = []
    for i in range(len(test_texts)):
        if predictions[i] != true_labels[i]:
            errors.append({
                'text': test_texts[i],
                'true': cefr_levels[true_labels[i]],
                'predicted': cefr_levels[predictions[i]],
                'index': i
            })

    print(f"Total misclassifications: {len(errors)}")

    if len(errors) > 0:
        print(
            f"\nShowing first {min(max_samples, len(errors))} misclassified samples:")
        for i, error in enumerate(errors[:max_samples]):
            text_preview = error['text'][:100] + \
                "..." if len(error['text']) > 100 else error['text']
            print(
                f"{i+1}. True: {error['true']} → Predicted: {error['predicted']}")
            print(f"   Text: \"{text_preview}\"")


def analyze_class_distribution(true_labels, predictions, cefr_levels):
    """Analyze class distribution in predictions vs true labels"""

    true_counts = np.bincount(true_labels, minlength=6)
    pred_counts = np.bincount(predictions, minlength=6)

    print("Class distribution comparison:")
    print(f"{'Level':<6} {'True':<8} {'Predicted':<10} {'Difference':<10}")
    print("-" * 40)

    for i, level in enumerate(cefr_levels):
        diff = pred_counts[i] - true_counts[i]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(
            f"{level:<6} {true_counts[i]:<8} {pred_counts[i]:<10} {diff_str:<10}")


def plot_confusion_matrix(cm, cefr_levels):
    """Plot confusion matrix heatmap"""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cefr_levels, yticklabels=cefr_levels)
        plt.title('Confusion Matrix - CEFR Level Classification')
        plt.xlabel('Predicted Level')
        plt.ylabel('True Level')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"⚠️ Could not save confusion matrix plot: {e}")


def plot_class_distribution(true_labels, predictions, cefr_levels):
    """Plot class distribution comparison"""
    try:
        true_counts = np.bincount(true_labels, minlength=6)
        pred_counts = np.bincount(predictions, minlength=6)

        x = np.arange(len(cefr_levels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, true_counts, width,
                       label='True Labels', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_counts, width,
                       label='Predictions', alpha=0.8)

        ax.set_xlabel('CEFR Levels')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution: True Labels vs Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(cefr_levels)
        ax.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Class distribution plot saved as 'class_distribution.png'")
    except Exception as e:
        print(f"⚠️ Could not save class distribution plot: {e}")


def main():
    print("🔄 Training CEFR BERT Model...")

    # Check if model already exists
    model_path = 'cefr_bert_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return

    # Initialize analyzer with classification settings
    analyzer = CEFRTextAnalyzer(
        model_name='bert-base-uncased',
        max_length=128,
        batch_size=4,
        learning_rate=2e-5,
        use_weighted_loss=True,  # Handle class imbalance
        alpha=0.5  # Loss weighting parameter
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
        train_data, val_data, epochs=1, best_model_path=model_path)

    print(f"✅ Best model saved as {model_path}")

    # Test the model on test set with detailed evaluation
    print("🧪 Testing model on test set...")
    test_accuracy, test_f1 = analyzer.evaluate_dataframe(test_data)
    print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
    print(f"🎯 Final Test F1 Score (Macro): {test_f1:.4f}")

    # Get detailed predictions for comprehensive analysis
    print("\n📊 Generating detailed evaluation reports...")
    test_texts = test_data['text'].tolist()
    test_labels = test_data['label'].map(analyzer.label_mapping).values
    test_loader = analyzer.create_data_loader(
        test_texts, test_labels, shuffle=False)

    # Get predictions with confidence scores
    accuracy, f1, predictions, true_labels = analyzer.evaluate_detailed(
        test_loader)

    # Generate comprehensive reports
    generate_detailed_reports(analyzer, predictions, true_labels, test_texts)

    print("🎉 Training and testing completed!")

    # Demo some predictions
    print("\n🔮 Sample Predictions:")
    demo_texts = [
        "I like cats.",
        "She has been working on this project for months.",
        "The implementation requires careful consideration of multiple factors.",
        "The epistemological framework underlying this methodology is complex."
    ]

    for text in demo_texts:
        level, confidence = analyzer.predict_text(text)
        print(f"'{text}' → {level} (conf: {confidence:.3f})")


if __name__ == "__main__":
    main()
