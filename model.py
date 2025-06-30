import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import warnings
import os
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
            'label': torch.tensor(label, dtype=torch.long)
        }


def mean_pooling(token_embeddings, attention_mask):
    """Apply mean pooling to get sentence embeddings"""
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CEFRClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=6, dropout_rate=0.3,
                 use_weighted_loss=True, class_weights=None):
        super(CEFRClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Initialize classifier weights
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.use_weighted_loss = use_weighted_loss
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        # Use mean pooling instead of [CLS] token
        last_hidden_state = outputs.last_hidden_state
        sentence_embeddings = mean_pooling(last_hidden_state, attention_mask)

        # Apply dropout and classification
        output = self.dropout(sentence_embeddings)
        logits = self.classifier(output)

        return logits


class CEFRTextAnalyzer:
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16,
                 learning_rate=2e-5, use_weighted_loss=True, alpha=0.5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_weighted_loss = use_weighted_loss
        self.alpha = alpha  # For loss weighting
        self.num_classes = 6  # A1, A2, B1, B2, C1, C2

        # Device setup - more conservative approach for stability
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(
                f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS can be unstable with some models, so let's be more cautious
            try:
                self.device = torch.device('mps')
                print("🍎 MPS (Apple Silicon) detected! Testing MPS...")
                # Test MPS with a small tensor operation
                test_tensor = torch.randn(2, 2).to(self.device)
                _ = test_tensor @ test_tensor  # Simple matrix multiplication test
                print("✅ MPS test passed - using MPS acceleration")
            except Exception as e:
                print(f"⚠️  MPS test failed: {e}")
                print("🔄 Falling back to CPU for stability")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU")

        # Initialize tokenizer
        print(f"📦 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        print(f"🤖 Loading model: {model_name}")
        self.model = CEFRClassifier(model_name, num_classes=self.num_classes)
        self.model.to(self.device)

        # Label mapping
        self.label_mapping = {'A1': 0, 'A2': 1,
                              'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
        self.reverse_label_mapping = {
            v: k for k, v in self.label_mapping.items()}

        print(f"✅ Model initialized on {self.device}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")

    def compute_class_weights(self, labels, epsilon=1e-5):
        """Compute class weights for handling class imbalance"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_ratios = counts / np.sum(counts)

        # Apply power scaling (similar to CEFR-SP approach)
        weights = np.power(class_ratios, self.alpha) / \
            np.sum(np.power(class_ratios, self.alpha))
        weights = weights / (class_ratios + epsilon)

        # Create full weight tensor for all classes
        full_weights = np.ones(self.num_classes)
        for i, label in enumerate(unique_labels):
            full_weights[label] = weights[i]

        return torch.FloatTensor(full_weights).to(self.device)

    def load_data(self, file_path):
        """Load data from CSV file with text,label format"""
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()

        # Convert CEFR labels to numerical values
        labels = df['label'].map(self.label_mapping).values

        return texts, labels

    def create_data_loader(self, texts, labels, shuffle=True):
        """Create DataLoader for training/validation"""
        dataset = CEFRDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, train_file, val_file, epochs=3, best_model_path=None):
        """Train the model with improved classification approach"""
        # Load data
        train_texts, train_labels = self.load_data(train_file)
        val_texts, val_labels = self.load_data(val_file)

        print(f"📊 Training samples: {len(train_texts)}")
        print(f"📊 Validation samples: {len(val_texts)}")

        # Compute class weights for handling imbalanced data
        class_weights = None
        if self.use_weighted_loss:
            class_weights = self.compute_class_weights(train_labels)
            print(f"📊 Class weights computed: {class_weights}")

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
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Loss function with optional class weights
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📈 Total training steps: {total_steps}")

        best_val_f1 = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            all_train_preds = []
            all_train_labels = []

            progress_bar = tqdm(
                train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

                # Backward pass
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                # Store predictions for F1 calculation
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                # Update progress bar
                current_acc = correct_predictions / total_predictions
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                # Clear cache periodically
                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Calculate training metrics
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            train_f1 = f1_score(
                all_train_labels, all_train_preds, average='macro')
            train_losses.append(avg_loss)

            # Validation phase
            val_accuracy, val_f1, val_preds, val_true = self.evaluate_detailed(
                val_loader)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  📉 Train Loss: {avg_loss:.4f}')
            print(
                f'  🎯 Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}')
            print(f'  ✅ Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')

            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                print(f'  🏆 New best validation F1: {best_val_f1:.4f}')
                if best_model_path:
                    self.save_model(best_model_path)

            print('-' * 50)

        print(f"🎉 Training completed! Best validation F1: {best_val_f1:.4f}")
        return train_losses, val_accuracies

    def train_from_dataframes(self, train_df, val_df, epochs=3, best_model_path=None):
        """Train the model from pandas DataFrames"""
        # Extract texts and labels from dataframes
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()

        train_labels = train_df['label'].map(self.label_mapping).values
        val_labels = val_df['label'].map(self.label_mapping).values

        print(f"📊 Training samples: {len(train_texts)}")
        print(f"📊 Validation samples: {len(val_texts)}")

        # Compute class weights
        class_weights = None
        if self.use_weighted_loss:
            class_weights = self.compute_class_weights(train_labels)
            print(f"📊 Class weights computed: {class_weights}")

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
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Loss function
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        print(f"🚀 Starting training for {epochs} epochs...")
        best_val_f1 = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            all_train_preds = []
            all_train_labels = []

            progress_bar = tqdm(
                train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                current_acc = correct_predictions / total_predictions
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Calculate metrics
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            train_f1 = f1_score(
                all_train_labels, all_train_preds, average='macro')

            # Validation
            val_accuracy, val_f1, _, _ = self.evaluate_detailed(val_loader)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  📉 Train Loss: {avg_loss:.4f}')
            print(
                f'  🎯 Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}')
            print(f'  ✅ Val Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                print(f'  🏆 New best validation F1: {best_val_f1:.4f}')
                if best_model_path:
                    self.save_model(best_model_path)

            print('-' * 50)

        print(f"🎉 Training completed! Best validation F1: {best_val_f1:.4f}")

    def evaluate(self, data_loader):
        """Simple evaluate method for backward compatibility"""
        accuracy, _, _, _ = self.evaluate_detailed(data_loader)
        return accuracy

    def evaluate_detailed(self, data_loader):
        """Detailed evaluation with accuracy, F1, predictions and true labels"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(
                    self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                logits = self.model(input_ids, attention_mask)
                _, preds = torch.max(logits, dim=1)

                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                all_predictions.extend(preds.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        accuracy = correct_predictions / total_predictions
        f1 = f1_score(all_true_labels, all_predictions, average='macro')

        return accuracy.item(), f1, all_predictions, all_true_labels

    def test(self, test_file):
        """Test the model and generate detailed results"""
        test_texts, test_labels = self.load_data(test_file)
        test_loader = self.create_data_loader(
            test_texts, test_labels, shuffle=False)

        print("Testing model...")
        accuracy, f1, predictions, true_labels = self.evaluate_detailed(
            test_loader)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score (Macro): {f1:.4f}")

        # Classification report
        cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions,
                                    labels=list(range(6)), target_names=cefr_levels,
                                    digits=4, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=list(range(6)))
        print(f"\nConfusion Matrix:")
        print("      ", " ".join(f"{level:>4}" for level in cefr_levels))
        for i, level in enumerate(cefr_levels):
            print(f"{level}: ", " ".join(f"{cm[i][j]:>4}" for j in range(6)))

        return accuracy, f1, predictions, true_labels

    def predict_text(self, text):
        """Predict CEFR level for a single text"""
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
            logits = self.model(input_ids, attention_mask)

            # Get probabilities and prediction
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities.max().item()
            _, prediction = torch.max(logits, dim=1)

        predicted_level = self.reverse_label_mapping[prediction.item()]
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
                logits = self.model(input_ids, attention_mask)

                # Get probabilities and predictions
                probabilities = torch.softmax(logits, dim=1)
                confidences = probabilities.max(dim=1)[0].cpu().numpy()
                predictions = torch.max(logits, dim=1)[1].cpu().numpy()

                # Convert to CEFR levels
                for j, (pred, conf) in enumerate(zip(predictions, confidences)):
                    results.append({
                        'sentence': batch_texts[j],
                        'cefr': self.reverse_label_mapping[pred],
                        'confidence': float(conf)
                    })

        return results

    def save_model(self, path):
        """Save the trained model with metadata"""
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'use_weighted_loss': self.use_weighted_loss,
            'alpha': self.alpha,
            'num_classes': self.num_classes,
            'label_mapping': self.label_mapping,
            'device_type': self.device.type,
            'torch_version': torch.__version__,
        }

        torch.save(model_info, path)
        print(f"💾 Model saved to {path}")

        # Model size info
        model_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"📊 Model size: {model_size:.1f} MB")

    def load_model(self, path):
        """Load a trained model"""
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load configuration if available
            if 'max_length' in checkpoint:
                self.max_length = checkpoint['max_length']
            if 'batch_size' in checkpoint:
                self.batch_size = checkpoint['batch_size']
            if 'use_weighted_loss' in checkpoint:
                self.use_weighted_loss = checkpoint['use_weighted_loss']
            if 'alpha' in checkpoint:
                self.alpha = checkpoint['alpha']

            print(f"✅ Model loaded from {path}")

            if 'torch_version' in checkpoint:
                print(
                    f"📦 Model trained with PyTorch {checkpoint['torch_version']}")
            if 'device_type' in checkpoint:
                print(
                    f"🔧 Model trained on {checkpoint['device_type']}, loading on {self.device.type}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try loading without weights_only for compatibility
            try:
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from {path} (compatibility mode)")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                raise

    def predictions_to_cefr(self, predictions):
        """Convert numerical predictions (0-5) to CEFR labels"""
        return [self.reverse_label_mapping[pred] for pred in predictions]

    def cefr_to_predictions(self, cefr_labels):
        """Convert CEFR labels to numerical predictions (0-5)"""
        return [self.label_mapping[label] for label in cefr_labels]

    def evaluate_dataframe(self, df):
        """Evaluate the model on a DataFrame"""
        texts = df['text'].tolist()
        labels = df['label'].map(self.label_mapping).values

        data_loader = self.create_data_loader(texts, labels, shuffle=False)
        accuracy, f1, _, _ = self.evaluate_detailed(data_loader)

        return accuracy, f1
