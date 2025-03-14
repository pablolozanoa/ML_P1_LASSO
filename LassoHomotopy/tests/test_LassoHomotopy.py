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
    lambda_value = 0.5 if "small_test" in filename else 1.5
    model = LassoHomotopyModel(lambda_init=lambda_value, eta=0.005)
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
    model = LassoHomotopyModel(lambda_init=1.5, eta=0.01, max_iter=300)
    print(model.theta)
    results = model.fit(X, y)
    preds = results.predict(X)
    mse = mean_squared_error_manual(y, preds)
    
    print(f"Testing with collinear data (correlation={correlation})")
    print(f"Mean Squared Error: {mse:.4f}")
    
    nonzero_count = np.count_nonzero(results.theta)
    print(f"Number of nonzero coefficients: {nonzero_count}")
    
    plot_coefficients(results.theta, f"LassoHomotopy Coefficients - Correlation {correlation}")
    
    assert mse < 15, "MSE is too high for collinear data"
    assert nonzero_count <= X.shape[1] / 2, "Model is not producing sparse coefficients as expected"

def test_lambda_effect():
    """Test to check the effect of lambda on sparsity"""
    X, y = generate_collinear_data(n_samples=50, n_features=20, correlation=0.98)
    
    model_low_lambda = LassoHomotopyModel(lambda_init=0.01, max_iter=200)
    model_high_lambda = LassoHomotopyModel(lambda_init=5.0, max_iter=200)
    
    results_low = model_low_lambda.fit(X, y)
    results_high = model_high_lambda.fit(X, y)
    
    nonzero_low = np.count_nonzero(results_low.theta)
    nonzero_high = np.count_nonzero(results_high.theta)
    
    print("\n===== Lambda Effect Test =====")
    print(f"Non-zero coefficients with low lambda ({model_low_lambda.lambda_reg}): {nonzero_low}")
    print(f"Non-zero coefficients with high lambda ({model_high_lambda.lambda_reg}): {nonzero_high}")
    print("================================")
    
    assert nonzero_high < nonzero_low, "Higher lambda should lead to sparser solution"

def test_zero_lambda():
    """If lambda is zero, solution should be OLS"""
    X, y = generate_collinear_data(n_samples=50, n_features=10)
    
    model = LassoHomotopyModel(lambda_init=0.0, max_iter=200)
    results = model.fit(X, y)
    
    nonzero_coeffs = np.count_nonzero(results.theta)
    
    print("\n===== Zero Lambda Test =====")
    print(f"Non-zero coefficients (expected full OLS solution): {nonzero_coeffs}")
    print(f"Theta values: {results.theta}")
    print("================================")
    
    assert nonzero_coeffs == X.shape[1], "Lambda=0 should not lead to sparsity"

def test_highly_correlated_features():
    """Test model performance on highly collinear data"""
    X, y = generate_collinear_data(n_samples=100, n_features=10, correlation=0.99)
    
    model = LassoHomotopyModel(lambda_init=1.0, eta=0.01, max_iter=200)
    results = model.fit(X, y)
    
    nonzero_count = np.count_nonzero(results.theta)
    
    print("\n===== Collinear Data Test =====")
    print(f"Number of nonzero coefficients: {nonzero_count}")
    print(f"Theta values: {results.theta}")
    print("================================")
    
    assert nonzero_count <= X.shape[1] // 2, "Model should enforce sparsity on collinear data"
