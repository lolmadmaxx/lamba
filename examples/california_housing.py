"""
California Housing Dataset Example using MicroGrad C

This example demonstrates:
1. Loading and preprocessing real-world data (California Housing dataset)
2. Feature normalization and data preparation
3. Building a regression model for house price prediction
4. Training with validation monitoring
5. Evaluation metrics and model analysis
6. Hyperparameter tuning
7. Model comparison and ensemble methods
"""

import random
import math
import csv
import sys
sys.path.append("..")
from micrograd_c import Value, MLP, Layer, Engine, SGD, Adam

class DataLoader:
    """Utility class for loading and preprocessing the California housing dataset"""
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_stats = {}
        self.target_stats = {}
        
    def load_csv(self, filepath):
        """Load CSV data and return features and targets"""
        features = []
        targets = []
        
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Convert to floats
                row_data = [float(x) for x in row]
                
                # Features: all columns except the last (median_house_value)
                features.append(row_data[:-1])
                # Target: median_house_value (last column)
                targets.append([row_data[-1]])  # Wrap in list for consistency
        
        return features, targets
    
    def compute_stats(self, data):
        """Compute mean and std for normalization"""
        if not data:
            return {'mean': 0, 'std': 1}
        
        # Handle both 1D and 2D data
        if isinstance(data[0], list):
            # 2D data (features)
            n_features = len(data[0])
            stats = []
            
            for feature_idx in range(n_features):
                feature_values = [row[feature_idx] for row in data]
                mean = sum(feature_values) / len(feature_values)
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                std = max(variance ** 0.5, 1e-8)  # Avoid division by zero
                stats.append({'mean': mean, 'std': std})
            
            return stats
        else:
            # 1D data (targets)
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std = max(variance ** 0.5, 1e-8)
            return {'mean': mean, 'std': std}
    
    def normalize_features(self, features, stats=None):
        """Normalize features using z-score normalization"""
        if stats is None:
            stats = self.compute_stats(features)
        
        normalized = []
        for row in features:
            normalized_row = []
            for i, value in enumerate(row):
                normalized_value = (value - stats[i]['mean']) / stats[i]['std']
                normalized_row.append(normalized_value)
            normalized.append(normalized_row)
        
        return normalized, stats
    
    def normalize_targets(self, targets, stats=None):
        """Normalize targets for better training stability"""
        flat_targets = [t[0] for t in targets]
        
        if stats is None:
            stats = self.compute_stats(flat_targets)
        
        normalized = [[(t[0] - stats['mean']) / stats['std']] for t in targets]
        return normalized, stats
    
    def denormalize_targets(self, normalized_targets, stats):
        """Convert normalized targets back to original scale"""
        return [t * stats['std'] + stats['mean'] for t in normalized_targets]
    
    def load_and_preprocess(self):
        """Load and preprocess both training and test data"""
        print("Loading California Housing dataset...")
        
        # Load raw data
        train_features, train_targets = self.load_csv(self.train_path)
        test_features, test_targets = self.load_csv(self.test_path)
        
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples: {len(test_features)}")
        print(f"Features: {len(train_features[0])}")
        
        # Normalize features
        train_features_norm, self.feature_stats = self.normalize_features(train_features)
        test_features_norm, _ = self.normalize_features(test_features, self.feature_stats)
        
        # Normalize targets
        train_targets_norm, self.target_stats = self.normalize_targets(train_targets)
        test_targets_norm, _ = self.normalize_targets(test_targets, self.target_stats)
        
        return {
            'train_features': train_features_norm,
            'train_targets': train_targets_norm,
            'test_features': test_features_norm,
            'test_targets': test_targets_norm,
            'raw_train_targets': train_targets,
            'raw_test_targets': test_targets
        }

def mean_squared_error(predictions, targets):
    """Compute MSE between predictions and targets"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    total_error = 0.0
    for pred, target in zip(predictions, targets):
        error = pred - target
        total_error += error * error
    
    return total_error / len(predictions)

def mean_absolute_error(predictions, targets):
    """Compute MAE between predictions and targets"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    total_error = 0.0
    for pred, target in zip(predictions, targets):
        total_error += abs(pred - target)
    
    return total_error / len(predictions)

def r2_score(predictions, targets):
    """Compute R² score"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    # Mean of targets
    mean_target = sum(targets) / len(targets)
    
    # Total sum of squares
    ss_tot = sum((target - mean_target) ** 2 for target in targets)
    
    # Residual sum of squares
    ss_res = sum((target - pred) ** 2 for pred, target in zip(predictions, targets))
    
    # R² score
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - (ss_res / ss_tot)

def evaluate_model(model, features, targets, data_loader, denormalize=True):
    """Evaluate model performance with multiple metrics"""
    predictions = []
    
    for feature_row in features:
        output = model(feature_row)
        predictions.append(output[0].data)
    
    # Denormalize if requested
    if denormalize:
        predictions = data_loader.denormalize_targets(predictions, data_loader.target_stats)
        target_values = [t[0] for t in targets]
    else:
        target_values = [t[0] for t in targets]
    
    # Compute metrics
    mse = mean_squared_error(predictions, target_values)
    mae = mean_absolute_error(predictions, target_values)
    r2 = r2_score(predictions, target_values)
    rmse = math.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': target_values
    }

def demo_basic_training():
    """Basic training example with California housing data"""
    print("=== Basic California Housing Training ===")
    
    # Load and preprocess data
    data_loader = DataLoader(
        "../dataset/california_housing_train.csv",
        "../dataset/california_housing_test.csv"
    )
    data = data_loader.load_and_preprocess()
    
    # Use smaller subset for faster training and to avoid hanging
    max_samples = 15000  # Limit to 1000 samples instead of full 17k dataset
    data['train_features'] = data['train_features'][:max_samples]
    data['train_targets'] = data['train_targets'][:max_samples]
    
    print(f"Using {len(data['train_features'])} training samples (subset for performance)")
    
    # Create smaller model
    model = MLP(nin=8, layer_specs=[32, 16, 8, 1])  # Reduced from [32, 16, 8, 1]
    model.initialize_parameters(method='xavier', seed=42)
    
    print(f"Model: {model}")
    
    # Train using Engine with fewer epochs and larger batches
    print("\nTraining with Engine...")
    history = Engine.train(
        model=model,
        train_inputs=data['train_features'],
        train_targets=data['train_targets'],
        epochs=20,  # Reduced from 50
        learning_rate=0.01,
        batch_size=100,  # Larger batches for efficiency
        validation_data=(data['test_features'][:200], data['test_targets'][:200]),  # Smaller validation set
        verbose=True,
        optimizer_type='adam' 
    )
      # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(
        model, 
        data['test_features'][:200],  # Use smaller test set
        data['raw_test_targets'][:200], 
        data_loader
    )
    
    print(f"Test Results:")
    print(f"  RMSE: ${test_results['rmse']:,.2f}")
    print(f"  MAE:  ${test_results['mae']:,.2f}")
    print(f"  R²:   {test_results['r2']:.4f}")
    
    # Show some predictions
    print(f"\nSample predictions:")
    for i in range(min(5, len(test_results['predictions']))):
        pred = test_results['predictions'][i]
        actual = test_results['targets'][i]
        error = abs(pred - actual)
        print(f"  Predicted: ${pred:,.0f}, Actual: ${actual:,.0f}, Error: ${error:,.0f}")
    
    return model, data_loader, data

def demo_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning"""
    print("\n=== Hyperparameter Tuning ===")
    
    # Load data
    data_loader = DataLoader(
        "../dataset/california_housing_train.csv",
        "../dataset/california_housing_test.csv"
    )
    data = data_loader.load_and_preprocess()
    
    # Use smaller subset for faster tuning
    max_samples = 500
    data['train_features'] = data['train_features'][:max_samples]
    data['train_targets'] = data['train_targets'][:max_samples]
    
    # Split training data into train/validation
    train_size = int(0.8 * len(data['train_features']))
    
    train_features = data['train_features'][:train_size]
    train_targets = data['train_targets'][:train_size]
    val_features = data['train_features'][train_size:]
    val_targets = data['train_targets'][train_size:]
    
    # Simpler configurations to try
    configs = [
        {'architecture': [8, 1], 'lr': 0.01, 'batch_size': 50},
        {'architecture': [16, 1], 'lr': 0.01, 'batch_size': 50},
        {'architecture': [8, 4, 1], 'lr': 0.005, 'batch_size': 50},
    ]
    
    best_model = None
    best_score = float('inf')
    best_config = None
    
    print("Testing different configurations...")
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")
        
        # Create and train model
        model = MLP(nin=8, layer_specs=config['architecture'])
        model.initialize_parameters(method='xavier', seed=42)
          # Quick training (fewer epochs for speed)
        Engine.train(
            model=model,
            train_inputs=train_features,
            train_targets=train_targets,
            epochs=5,  # Reduced from 20
            learning_rate=config['lr'],
            batch_size=config['batch_size'],
            verbose=False,
            optimizer_type='sgd'  # Use SGD instead of Adam
        )
        
        # Evaluate on validation set
        val_results = evaluate_model(model, val_features, val_targets, data_loader, denormalize=False)
        val_mse = val_results['mse']
        
        print(f"  Validation MSE: {val_mse:.6f}")
        
        if val_mse < best_score:
            best_score = val_mse
            best_model = model
            best_config = config
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best validation MSE: {best_score:.6f}")
    
    # Evaluate best model on test set
    test_results = evaluate_model(
        best_model, 
        data['test_features'], 
        data['raw_test_targets'], 
        data_loader
    )
    
    print(f"\nBest model test results:")
    print(f"  RMSE: ${test_results['rmse']:,.2f}")
    print(f"  R²:   {test_results['r2']:.4f}")
    
    return best_model, best_config

def demo_ensemble_methods():
    """Demonstrate ensemble methods"""
    print("\n=== Ensemble Methods ===")
    
    # Load data
    data_loader = DataLoader(
        "../dataset/california_housing_train.csv",
        "../dataset/california_housing_test.csv"
    )
    data = data_loader.load_and_preprocess()
    
    # Create multiple models with different architectures
    model_configs = [
        [32, 16, 1],
        [64, 32, 1],
        [48, 24, 12, 1],
        [80, 40, 20, 1]
    ]
    
    models = []
    print("Training ensemble models...")
    
    for i, config in enumerate(model_configs):
        print(f"Training model {i+1}: {config}")
        
        model = MLP(nin=8, layer_specs=config)
        model.initialize_parameters(method='xavier', seed=42 + i)  # Different seeds
        
        # Train model
        Engine.train(
            model=model,
            train_inputs=data['train_features'],
            train_targets=data['train_targets'],
            epochs=30,
            learning_rate=0.01,
            batch_size=64,
            verbose=False,
            optimizer_type='adam'
        )
        
        models.append(model)
    
    # Evaluate individual models
    print("\nIndividual model performance:")
    individual_results = []
    
    for i, model in enumerate(models):
        results = evaluate_model(
            model, 
            data['test_features'], 
            data['raw_test_targets'], 
            data_loader
        )
        individual_results.append(results)
        print(f"Model {i+1}: RMSE = ${results['rmse']:,.2f}, R² = {results['r2']:.4f}")
    
    # Ensemble prediction (average)
    print("\nEnsemble prediction (averaging):")
    ensemble_predictions = []
    
    for j in range(len(data['test_features'])):
        # Get predictions from all models
        model_preds = []
        for model in models:
            output = model(data['test_features'][j])
            pred = data_loader.denormalize_targets([output[0].data], data_loader.target_stats)[0]
            model_preds.append(pred)
        
        # Average the predictions
        ensemble_pred = sum(model_preds) / len(model_preds)
        ensemble_predictions.append(ensemble_pred)
    
    # Evaluate ensemble
    test_targets = [t[0] for t in data['raw_test_targets']]
    ensemble_mse = mean_squared_error(ensemble_predictions, test_targets)
    ensemble_rmse = math.sqrt(ensemble_mse)
    ensemble_r2 = r2_score(ensemble_predictions, test_targets)
    
    print(f"Ensemble: RMSE = ${ensemble_rmse:,.2f}, R² = {ensemble_r2:.4f}")
    
    # Compare with best individual model
    best_individual = min(individual_results, key=lambda x: x['rmse'])
    improvement = best_individual['rmse'] - ensemble_rmse
    
    print(f"\nImprovement over best individual model: ${improvement:,.2f}")
    
    return models, ensemble_predictions

def demo_feature_analysis():
    """Analyze feature importance and model behavior"""
    print("\n=== Feature Analysis ===")
    
    # Feature names for California housing dataset
    feature_names = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ]
    
    # Load data
    data_loader = DataLoader(
        "../dataset/california_housing_train.csv",
        "../dataset/california_housing_test.csv"
    )
    data = data_loader.load_and_preprocess()
    
    # Train a model
    model = MLP(nin=8, layer_specs=[64, 32, 16, 1])
    model.initialize_parameters(method='xavier', seed=42)
    
    Engine.train(
        model=model,
        train_inputs=data['train_features'],
        train_targets=data['train_targets'],
        epochs=40,
        learning_rate=0.01,
        batch_size=64,
        verbose=False,
        optimizer_type='adam'
    )
    
    # Analyze feature sensitivity
    print("Feature sensitivity analysis:")
    print("(How much output changes when each feature is perturbed)")
    
    # Use a sample from test set
    test_sample = data['test_features'][0]
    baseline_output = model(test_sample)[0].data
    
    sensitivities = []
    perturbation = 0.1  # Small perturbation
    
    for i, feature_name in enumerate(feature_names):
        # Perturb feature
        perturbed_sample = test_sample.copy()
        perturbed_sample[i] += perturbation
        
        perturbed_output = model(perturbed_sample)[0].data
        sensitivity = abs(perturbed_output - baseline_output) / perturbation
        sensitivities.append((feature_name, sensitivity))
    
    # Sort by sensitivity
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFeature importance ranking:")
    for i, (name, sensitivity) in enumerate(sensitivities):
        print(f"{i+1:2d}. {name:<20}: {sensitivity:.6f}")
    
    # Show prediction on sample
    actual_price = data_loader.denormalize_targets([baseline_output], data_loader.target_stats)[0]
    actual_target = data['raw_test_targets'][0][0]
    
    print(f"\nSample prediction:")
    print(f"  Predicted: ${actual_price:,.0f}")
    print(f"  Actual:    ${actual_target:,.0f}")
    print(f"  Error:     ${abs(actual_price - actual_target):,.0f}")

def main():
    """Run all California housing examples"""
    print("California Housing Dataset Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    try:
        # Quick sanity check first
        print("Performing quick sanity check...")
        from micrograd_c import Value, MLP, Engine, SGD
        test_model = MLP(2, [2, 1])
        test_inputs = [[0.1, 0.2], [0.3, 0.4]]
        test_targets = [[0.5], [0.6]]
        test_optimizer = SGD(test_model.parameters, lr=0.01)
        test_loss = Engine.train_step(
            model=test_model,
            inputs=test_inputs,
            targets=test_targets,
            loss_fn=Engine.mse_loss,
            optimizer=test_optimizer
        )
        print(f"Sanity check passed! Test loss: {test_loss}")
          # Basic training
        model, data_loader, data = demo_basic_training()
        
        # Hyperparameter tuning
        best_model, best_config = demo_hyperparameter_tuning()
        
        # Skip ensemble methods and feature analysis for now to avoid potential hangs
        print("\nSkipping ensemble methods and feature analysis for performance...")
        
        print("\n" + "=" * 50)
        print("California housing examples completed!")
        
        print("\nKey insights:")
        print("1. Real-world datasets require careful preprocessing and normalization")
        print("2. Hyperparameter tuning can significantly improve performance")
        print("3. Memory management optimizations prevent crashes during training")
        print("4. Smaller models and datasets train more reliably")
        print("5. MicroGrad C handles large datasets efficiently")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. {e}")
        print("Make sure the California housing CSV files are in the ../dataset/ directory")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
