import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import os
from accelerate import Accelerator
warnings.filterwarnings('ignore')


class CEFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CEFRClassifier(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=6, dropout_rate=0.3):
        super(CEFRClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class CEFRTextAnalyzer:
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16, learning_rate=2e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Improved device detection and setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(
                f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Clear cache for better memory management
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🍎 MPS (Apple Silicon) detected! Using MPS acceleration")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU (consider using GPU for faster training)")

        # Initialize tokenizer and model
        print(f"📦 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"🤖 Loading model: {model_name}")
        self.model = CEFRClassifier(model_name)
        self.model.to(self.device)

        # Enable mixed precision for GPU
        self.use_mixed_precision = self.device.type in ['cuda', 'mps']
        if self.use_mixed_precision:
            print("⚡ Mixed precision training enabled for faster training")

        print(f"✅ Model initialized on {self.device}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")

    def load_data(self, file_path):
        """Load data from CSV file with text,label format"""
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()

        # Map CEFR labels to numerical values (0-5)
        label_mapping = {
            'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5
        }

        # Convert string labels to numerical labels
        labels = df['label'].map(label_mapping).values

        return texts, labels

    def create_data_loader(self, texts, labels, shuffle=True):
        """Create DataLoader for training/validation"""
        dataset = CEFRDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, train_file, val_file, epochs=3, best_model_path=None):
        """Train the model with improved GPU/CPU handling"""
        # Load data
        train_texts, train_labels = self.load_data(train_file)
        val_texts, val_labels = self.load_data(val_file)

        print(f"📊 Training samples: {len(train_texts)}")
        print(f"📊 Validation samples: {len(val_texts)}")

        # Create data loaders
        train_loader = self.create_data_loader(
            train_texts, train_labels, shuffle=True)
        val_loader = self.create_data_loader(
            val_texts, val_labels, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(),
                          lr=self.learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        # Mixed precision scaler for GPU
        scaler = torch.amp.GradScaler() if self.use_mixed_precision else None

        # Training loop
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📈 Total training steps: {total_steps}")

        best_val_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(
                train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass with mixed precision if available
                if self.use_mixed_precision and scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(input_ids, attention_mask)
                        loss = loss_fn(outputs, labels)

                    # Backward pass with scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                # Update progress bar
                current_acc = correct_predictions / total_predictions
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                # Clear cache periodically for GPU
                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Validation
            val_accuracy = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  📉 Train Loss: {avg_loss:.4f}')
            print(f'  🎯 Train Accuracy: {train_accuracy:.4f}')
            print(f'  ✅ Validation Accuracy: {val_accuracy:.4f}')

            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                print(f'  🏆 New best validation accuracy: {best_val_acc:.4f}')
                if best_model_path:
                    self.save_model(best_model_path)

            print('-' * 50)

        print(
            f"🎉 Training completed! Best validation accuracy: {best_val_acc:.4f}")

    def evaluate(self, data_loader):
        """Evaluate the model with improved efficiency"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                # Use mixed precision for inference if available
                if self.use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        return correct_predictions / total_predictions

    def test(self, test_file):
        """Test the model and generate detailed results"""
        test_texts, test_labels = self.load_data(test_file)
        test_loader = self.create_data_loader(
            test_texts, test_labels, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        print("Testing model...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Keep in 0-5 range for internal calculations
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Classification report
        cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        labels = list(range(6))  # 0-5 for CEFR levels
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions,
                                    labels=labels, target_names=cefr_levels,
                                    digits=4, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        print(f"\nConfusion Matrix:")
        print(cm)

        return accuracy, predictions, true_labels

    def predict_text(self, text):
        """Predict CEFR level for a single text with improved efficiency"""
        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device, non_blocking=True)
        attention_mask = encoding['attention_mask'].to(
            self.device, non_blocking=True)

        with torch.no_grad():
            # Use mixed precision for inference if available
            if self.use_mixed_precision:
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(input_ids, attention_mask)
            else:
                outputs = self.model(input_ids, attention_mask)

            # Get probabilities and prediction
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities.max().item()
            _, prediction = torch.max(outputs, dim=1)

        cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        predicted_level = cefr_levels[prediction.item()]

        return predicted_level, confidence

    def predict_batch(self, texts):
        """Predict CEFR levels for multiple texts efficiently"""
        self.model.eval()

        results = []

        # Process in batches for efficiency
        batch_size = min(self.batch_size, len(texts))

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = encodings['attention_mask'].to(
                    self.device, non_blocking=True)

                # Predict
                if self.use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask)

                # Get probabilities and predictions
                probabilities = torch.softmax(outputs, dim=1)
                confidences = probabilities.max(dim=1)[0].cpu().numpy()
                predictions = torch.max(outputs, dim=1)[1].cpu().numpy()

                # Convert to CEFR levels
                cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
                for j, (pred, conf) in enumerate(zip(predictions, confidences)):
                    results.append({
                        'sentence': batch_texts[j],
                        'cefr': cefr_levels[pred],
                        'confidence': float(conf)
                    })

        return results

    def save_model(self, path):
        """Save the trained model with improved metadata"""
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device_type': self.device.type,
            'torch_version': torch.__version__,
            'transformers_version': getattr(__import__('transformers'), '__version__', 'unknown')
        }

        torch.save(model_info, path)
        print(f"💾 Model saved to {path}")

        # Save model size info
        model_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"📊 Model size: {model_size:.1f} MB")

    def load_model(self, path):
        """Load a trained model with compatibility checks"""
        try:
            # Try loading with weights_only=True first (safer)
            try:
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=True)
            except Exception as e:
                # If weights_only=True fails, try with weights_only=False
                print(
                    f"⚠️  Warning: Loading with weights_only=False due to compatibility issue")
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load additional info if available
            if 'max_length' in checkpoint:
                self.max_length = checkpoint['max_length']
            if 'batch_size' in checkpoint:
                self.batch_size = checkpoint['batch_size']

            print(f"✅ Model loaded from {path}")

            # Show compatibility info
            if 'torch_version' in checkpoint:
                print(
                    f"📦 Model trained with PyTorch {checkpoint['torch_version']}")
            if 'device_type' in checkpoint:
                print(
                    f"🔧 Model trained on {checkpoint['device_type']}, loading on {self.device.type}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def predictions_to_cefr(self, predictions):
        """Convert numerical predictions (0-5) to CEFR labels"""
        cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        return [cefr_levels[pred] for pred in predictions]

    def cefr_to_predictions(self, cefr_labels):
        """Convert CEFR labels to numerical predictions (0-5)"""
        label_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
        return [label_mapping[label] for label in cefr_labels]


def main():
    # Initialize the analyzer
    analyzer = CEFRTextAnalyzer(
        model_name='bert-base-uncased',
        max_length=128,
        batch_size=8,  # Smaller batch size for stability
        learning_rate=2e-5
    )

    # Define file paths
    train_file = 'dataset/train.csv'
    val_file = 'dataset/validation.csv'
    test_file = 'dataset/test.csv'

    # Train the model
    analyzer.train(train_file, val_file, epochs=5)

    # Test the model
    accuracy, predictions, true_labels = analyzer.test(test_file)

    # Save the model
    analyzer.save_model('cefr_bert_model.pth')

    # Example predictions
    print("\n" + "="*50)
    print("Example Predictions:")
    print("="*50)

    example_texts = [
        "I like cats.",
        "The weather is nice today.",
        "She has been working on this project for several months.",
        "The implementation requires careful consideration of multiple factors.",
        "The sophisticated methodology demonstrates exceptional analytical rigor.",
        "The paradigmatic shift necessitates comprehensive epistemological examination."
    ]

    for text in example_texts:
        level, confidence = analyzer.predict_text(text)
        print(f"Text: '{text}'")
        print(f"Predicted CEFR Level: {level} (Confidence: {confidence:.3f})")
        print("-" * 50)


if __name__ == "__main__":
    main()
