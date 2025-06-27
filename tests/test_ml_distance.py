from kmerml.ml.features import KmerFeatureBuilder
from kmerml.ml.distance_learning import PhylogeneticDistancePredictor, prepare_distance_data
from kmerml.ml.feature_selection import select_features_by_variance
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

# Create output directories
Path("data/models").mkdir(parents=True, exist_ok=True)
Path("data/results").mkdir(parents=True, exist_ok=True)

# 1. Load the feature matrix with filtering for k≥8
print("Loading feature matrix...")
builder = KmerFeatureBuilder(stats_dir="data/processed/features/")
feature_matrix = builder.build_from_statistics_files(
    metric="shannon_entropy")
print(f"Feature matrix shape: {feature_matrix.shape}")

# 2. Apply feature selection (reduce dimensionality)
if feature_matrix.shape[1] > 1000:
    print("Selecting top 1,000 features by variance for faster testing")
    variance_features = select_features_by_variance(feature_matrix, n_features=1000)
    feature_matrix = feature_matrix[variance_features]
    print(f"Reduced feature matrix shape: {feature_matrix.shape}")

# 3. Generate reference distances
print("\nGenerating reference distances...")
dist_matrix = squareform(pdist(feature_matrix.values, metric='euclidean'))
print(f"Distance matrix shape: {dist_matrix.shape}")

# 4. Prepare data for distance learning
print("\nPreparing data for distance learning...")
X_pairs, y_pairs = prepare_distance_data(feature_matrix, dist_matrix)
print(f"X_pairs shape: {X_pairs.shape}, y_pairs shape: {y_pairs.shape}")

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pairs, y_pairs, test_size=0.2, random_state=42
)
print(f"Training data: {X_train.shape[0]} pairs, Testing data: {X_test.shape[0]} pairs")

# 6. Train and evaluate models
models = {
    'random_forest': PhylogeneticDistancePredictor(model_type='random_forest', n_estimators=100),
    'gradient_boosting': PhylogeneticDistancePredictor(model_type='gradient_boosting', n_estimators=100)
}

results = {}
print("\nTraining and evaluating models...")

for name, model in models.items():
    print(f"\nTraining {name} model...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    
    # Perform cross-validation
    cv_metrics = model.cross_validate(X_train, y_train, cv=5)
    
    # Print results
    print(f"{name} model training completed in {training_time:.2f} seconds")
    print(f"Test metrics: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")
    print(f"CV metrics: RMSE = {cv_metrics['cv_rmse_mean']:.4f} ± {cv_metrics['cv_rmse_std']:.4f}")
    
    # Store results
    results[name] = {
        'training_time': training_time,
        'test_metrics': metrics,
        'cv_metrics': cv_metrics
    }

# 7. Save results to CSV
results_list = []
for name, result in results.items():
    results_list.append({
        'model': name,
        'training_time': result['training_time'],
        'test_rmse': result['test_metrics']['rmse'],
        'test_r2': result['test_metrics']['r2'],
        'cv_rmse_mean': result['cv_metrics']['cv_rmse_mean'],
        'cv_r2_mean': result['cv_metrics']['cv_r2_mean']
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('data/results/distance_learning_results.csv', index=False)
print(f"\nSaved results to 'data/results/distance_learning_results.csv'")

# 8. Save best model
best_model_name = min(results, key=lambda x: results[x]['test_metrics']['rmse'])
best_model = models[best_model_name]
best_model.save('data/models/best_distance_predictor.pkl')
print(f"\nSaved best model ({best_model_name}) to 'data/models/best_distance_predictor.pkl'")

print("\nDistance learning testing complete!")

try:
    import matplotlib.pyplot as plt
    
    # Plot actual vs predicted distances
    plt.figure(figsize=(12, 5))
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        plt.subplot(1, 2, i+1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual Distances')
        plt.ylabel('Predicted Distances')
        plt.title(f'{name.replace("_", " ").title()} Model')
        
    plt.tight_layout()
    plt.savefig('data/results/distance_prediction_plot.png')
    print("Saved visualization to 'data/results/distance_prediction_plot.png'")
except ImportError:
    print("Matplotlib not available. Skipping visualizations.")