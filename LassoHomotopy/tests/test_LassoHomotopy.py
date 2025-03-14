import csv
import numpy as np
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from model.LassoHomotopy import LassoHomotopyModel


# Set Seaborn style for better plots
sns.set_theme(style="whitegrid")

def load_csv(filename):
    """ Load dataset from CSV file """
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x') or k.startswith('X')] for datum in data])
    y = np.array([float(datum['y']) if 'y' in datum else float(datum['target']) for datum in data])
    return X, y

def mean_squared_error_manual(y_true, y_pred):
    """ Compute Mean Squared Error without using sklearn """
    return np.mean((y_true - y_pred) ** 2)

def generate_collinear_data(n_samples=100, n_features=10, correlation=0.95):
    """ Generate highly collinear synthetic data """
    np.random.seed(42)
    X_base = np.random.randn(n_samples, 1)  # Single independent feature
    noise = np.random.randn(n_samples, n_features) * (1 - correlation)
    X = correlation * X_base + noise  # Create correlated features
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    return X, y

def plot_coefficients(coefficients, title):
    """ Plot model coefficients to visualize sparsity with improved aesthetics """
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=np.arange(len(coefficients)), y=coefficients, hue=np.arange(len(coefficients)), palette="coolwarm", edgecolor='black', legend=False)
    
    # Add value labels to bars with padding
    y_offset = max(abs(coefficients)) * 0.1  # Dynamic padding based on max coefficient
    for i, coef in enumerate(coefficients):
        ax.text(i, coef + y_offset if coef >= 0 else coef - y_offset, f'{coef:.2f}',
                ha='center', va='bottom' if coef >= 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Coefficient Value", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')  # Add reference line at 0
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(min(coefficients) - y_offset * 2, max(coefficients) + y_offset * 2)  # Adjust margin
    plt.show()

@pytest.mark.parametrize("filename", ["small_test.csv", "collinear_data.csv"])
def test_predict(filename):
    """ Test the LassoHomotopyModel using different datasets """
    model = LassoHomotopyModel(lambda_init=0.1)
    X, y = load_csv(filename)
    
    print(f"Testing with dataset: {filename}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    results = model.fit(X, y)
    preds = results.predict(X)
    
    mse = mean_squared_error_manual(y, preds)
    print(f"Mean Squared Error for {filename}: {mse:.4f}")
    
    plot_coefficients(results.theta, f"LassoHomotopy Coefficients - {filename}")
    model.plot_lambda_history()
    model.plot_theta_history()
    
    assert preds.shape == y.shape, "Prediction shape must match ground truth"
    assert mse < 15, "MSE is too high, model might not be fitting properly"

@pytest.mark.parametrize("correlation", [0.9, 0.95, 0.99])
def test_collinear_data(correlation):
    """ Test LassoHomotopyModel with highly collinear data """
    X, y = generate_collinear_data(correlation=correlation)
    
    # Increase lambda_init to encourage sparsity
    model = LassoHomotopyModel(lambda_init=2.0)
    results = model.fit(X, y)
    preds = results.predict(X)
    mse = mean_squared_error_manual(y, preds)
    
    print(f"Testing with collinear data (correlation={correlation})")
    print(f"Mean Squared Error: {mse:.4f}")
    
    nonzero_count = np.count_nonzero(results.theta)
    print(f"Number of nonzero coefficients: {nonzero_count}")
    
    plot_coefficients(results.theta, f"LassoHomotopy Coefficients - Correlation {correlation}")
    model.plot_lambda_history()
    model.plot_theta_history()
    
    assert mse < 15, "MSE is too high for collinear data"
    assert nonzero_count <= X.shape[1] / 2 + 1, "Model is not producing sparse coefficients as expected"