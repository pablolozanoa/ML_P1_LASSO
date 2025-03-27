# Project 1: 
LASSO regularized regression model using the Homotopy Method

## Authors
- Pablo Lozano Arias    (A20599454)  
- Nicolás Rigau Sinca   (A20595377)

---

## Introduction
This project implements **a LASSO regularized regression model** using the **Homotopy Method**, an efficient approach for solving sparse linear regression problems. The goal is to develop an implementation **from first principles**, avoiding built-in LASSO solvers like `scikit-learn`. Instead, the model leverages **NumPy** for numerical computations and is specifically designed to handle **highly collinear datasets**.

The implementation is inspired by the paper **"An Homotopy Algorithm for the Lasso with Online Observations"**, which introduces an **online learning approach to LASSO**. While our current model solves the LASSO problem iteratively, it incorporates **homotopy-based updates** and **dynamic lambda adjustments**, aligning closely with the methodology described in the paper.

---

## Project Structure
The repository is organized as follows:

```plaintext
Project 1
│── README.md                             # Project documentation
│── requirements.txt                      # Required dependencies for the project
│── LassoHomotopy/                        # Main project directory
│   ├── model/                            # Contains the LASSO implementation
│   │   ├── LassoHomotopy.py              # Homotopy-based LASSO model
│   ├── tests/                            # Contains test scripts and datasets
│   │   ├── test_LassoHomotopy.py         # Unit tests for the LASSO model
│   │   ├── collinear_data.csv            # Dataset with highly collinear features
│   │   ├── small_test.csv                # Small dataset for validation
|   |   ├── synthetic_test_plots.ipynb    # Jupyter notebook with syntetic test plots
│── generate_regression_data.py           # Script to generate synthetic regression data
```

The repository is structured to ensure a clear separation between the implementation, testing, and dataset generation. The `LassoHomotopy/` directory contains the core implementation of the LASSO model, with `LassoHomotopy.py` handling the homotopy-based regression algorithm. The `tests/` folder includes unit tests and datasets to validate the model’s accuracy, sparsity, and performance on collinear data, using `test_LassoHomotopy.py` along with the datasets `collinear_data.csv` and `small_test.csv`. Additionally, `generate_regression_data.py` provides a script to create synthetic regression datasets, allowing flexibility for testing different scenarios. This structure makes it easy to navigate, extend, and validate the model while maintaining clarity between its components.

# 1. Paper Review: 
An Homotopy Algorithm for the Lasso with Online Observations

## 1.1. Introduction to LASSO
LASSO (Least Absolute Shrinkage and Selection Operator) is a regression technique that introduces an ℓ₁ penalty to induce **sparsity** in the coefficient vector θ.

### Objective
To fit a linear model by minimizing the squared error with a penalty:
```math
\min_{\theta} \frac{1}{2} \sum_{i=1}^{n} (x_i^T \theta - y_i)^2 + \mu_n \left\|\theta \right\|_1
```
Where $\mu_n$ is the regularization parameter. The solution θ tends to be **sparse**, meaning many coefficients are zero, which helps with feature selection.

---

## 1.2. Why Use the Homotopy Method?
Traditional solvers like *Coordinate Descent* or *Least Angle Regression (LARS)* can be computationally expensive.

### Advantages of the Homotopy Method
- **Leverages previous calculations**: If we already have the solution for a dataset, we can update it **efficiently** when adding new data points.
- **Avoids recomputation from scratch**: It follows a trajectory of solutions as new information is introduced.

---

## 1.3. Optimality Conditions for LASSO
Since the objective function is convex but non-differentiable (due to the ℓ₁ norm), the optimal solution is characterized by the **subdifferential** of $\left\|\theta \right\|_1$:
```math
X^T (X \theta - y) + \mu_n v = 0, \quad v \in \partial \left\|\theta \right\|_1
```
Where $v$ is a vector of subgradients.

By defining the **active set** (A), which consists of the variables with nonzero coefficients in $\theta$, we can rewrite the solution in a closed form for the active coefficients.

---

## 1.4. Proposed Homotopy Algorithm (RecLasso)
The algorithm has **two main steps** when a new data point $(y_{n+1}, x_{n+1})$ is introduced:

### 1.4.1 Step 1: Update the Regularization Parameter $\mu_n$
If we want to change $\mu_n$ to a new value $\mu_{n+1}$, we efficiently follow the LASSO solution path.

### 1.4.2 Step 2: Vary the Parameter \( t \) from 0 to 1
We define the following problem:
```math
\theta(t, \mu) = \arg \min_{\theta} \frac{1}{2} \left\|(X, t x_{n+1}) \theta - (y, t y_{n+1})\right\|\tfrac{2}{2} + \mu \left\|\theta \right\|_1

```
This parameter $t$ allows us to continuously update the solution as the new observation is added.

### 1.4.3 Computing Transition Points
- As $t$ increases, the solution $\theta(t)$ changes **smoothly** until a change in the active set occurs (a coefficient becomes zero, or a new coefficient becomes active).
- The next transition point is computed, where this change occurs, and the solution is updated.

# 2. Answer the following questions.

## 2.1. What does the model you have implemented do and when should it be used?
The **LassoHomotopyModel** implements **LASSO regression using the Homotopy Method**, which efficiently computes a sparse solution for linear regression problems with ℓ₁ regularization.

### **Functionality:**
- Finds the optimal coefficient vector **θ** by minimizing the **LASSO objective function**:
```math
  \min_{\theta} \frac{1}{2} \sum_{i=1}^{n} (x_i^T \theta - y_i)^2 + \lambda \left\|\theta \right\|_1
```
- Uses the **Homotopy method** to update the solution iteratively without recomputing from scratch.
- Automatically updates the **regularization parameter** $\lambda$ based on prediction error.

### **Relevant Code (Extracted from `LassoHomotopy.py`)**

#### **Initialization of the Model**

```python
def __init__(self, lambda_init=0.1, eta=0.01, max_iter=100, tol=1e-6):
    self.lambda_reg = lambda_init  # Regularization parameter
    self.eta = eta  # Learning rate for lambda update
    self.max_iter = max_iter  # Maximum number of iterations
    self.tol = tol  # Convergence tolerance
    self.theta = None  # Model coefficients
    self.active_set = set()  # Set of active features
    self.lambda_history = []  # Store lambda updates
    self.theta_history = []  # Store theta updates
```

Sets up the initial parameters for the model. It includes:

- Initial regularization value (λ).
- Learning rate for dynamically adjusting λ.
- Number of iterations and tolerance for stopping.
- Containers to track model coefficients and parameter history.


#### **Fitting the Model (Core of Homotopy Method)**

```python
def fit(self, X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, m = X.shape
    self.theta = np.zeros(m)
    self.active_set = set()

```

- Prepares the data and initializes parameters for optimization.
- Sets $\theta$ to a zero vector, indicating that no features are active at the start.


#### **Iterative Update and Feature Selection**

```python
for _ in range(self.max_iter):
    gradient = X.T @ (X @ self.theta - y)
    idx = np.argmax(np.abs(gradient))

    if idx not in self.active_set:
        self.active_set.add(idx)

    active_list = list(self.active_set)
    X_active = X[:, active_list]
```

At each iteration:
- Calculates the gradient to find the most impactful feature.
- Adds that feature to the active set if it's not already present.
- Extracts the subset of X corresponding to active features.


#### **Homotopy Update of Coefficients**

```python
if len(active_list) > 0:
    theta_active_prev = self.theta[active_list].copy()
    theta_active = self._homotopy_update(X_active, y, theta_active_prev)
    self.theta[active_list] = theta_active
```

- Applies the homotopy formula to update the coefficients of the selected (active) features.
- Uses the previous coefficients as a warm start.


#### **Removing Inactive Coefficients**

```python
transition_points = np.where(np.abs(theta_active) < self.tol)[0]
for t in transition_points:
    if active_list[t] in self.active_set:
        self.active_set.remove(active_list[t])
```

- If any updated coefficient becomes smaller than the tolerance, it's removed from the active set, enforcing sparsity.


#### **Adaptive λ Update & History Tracking**

```python
self.lambda_history.append(self.lambda_reg)
self.theta_history.append(self.theta.copy())

prediction_error = np.mean((X @ self.theta - y) ** 2)
self.lambda_reg *= np.exp(-self.eta * prediction_error)
```

- Saves λ and θ values for each iteration (useful for plotting).
- Updates λ using an exponential decay based on prediction error (dynamic regularization).


#### **Convergence Check**

```python
if np.linalg.norm(gradient, ord=np.inf) < self.tol:
    break
```

- Stops the iteration when the largest gradient component is below the threshold (model has converged).


#### **Homotopy Update Function**

```python
def _homotopy_update(self, X_active, y, theta_active_prev):
    try:
        inv_X = np.linalg.inv(X_active.T @ X_active)
    except np.linalg.LinAlgError:
        inv_X = np.linalg.pinv(X_active.T @ X_active)

    return inv_X @ (X_active.T @ y - self.lambda_reg * np.sign(theta_active_prev))
```

- Computes a closed-form update for θ using matrix inversion.
- If the matrix is not invertible, uses the pseudo-inverse for stability.


#### **Visualization Tools (Optional)**

```python
def plot_lambda_history(self): ...
def plot_theta_history(self): ...
```

- These methods help visualize how λ and θ evolve over time.
- Not core to the algorithm but helpful in analysis.


#### **Prediction Function**

```python
def predict(self, X):
    X = np.array(X, dtype=float)
    return X @ self.theta
```

- Applies the learned θ to new data for making predictions.

### **When should it be used?**
- **Feature selection**: Identifies the most important variables by enforcing sparsity in **θ**.
- **High-dimensional data**: Works well when there are **more features (m) than observations (n)**.
- **Collinear data**: Handles cases where features are highly correlated.
- **Online learning**: Efficient for scenarios where **data arrives sequentially** (e.g., real-time applications).

---

## 2.2. How did you test your model to determine if it is working reasonably correctly?

The model is tested using multiple test cases defined in **test_LassoHomotopy.py**, ensuring its correctness and performance.

### **Key Tests:**
#### **1. Basic Dataset Validation (test_predict)**
- Loads datasets from CSV files (e.g., `small_test.csv`, `collinear_data.csv`).
- Trains the model and evaluates its performance by computing the **Mean Squared Error (MSE)**.
- Plots the **coefficients and λ evolution**.

#### **2. Collinearity Test (test_collinear_data)**
- Generates synthetic datasets where features are highly correlated (correlation: 0.9, 0.95, 0.99).
- Ensures that the model produces **sparse coefficients**.
- Asserts that the **MSE is low (< 15)** and that **less than half the features are nonzero**.

#### **3. Effect of λ on Sparsity (test_lambda_effect)**
- Runs the model with **low** and **high** λ values.
- Checks that a **higher λ leads to fewer nonzero coefficients**, enforcing more sparsity.

#### **4. Zero Regularization Case (test_zero_lambda)**
- Tests if setting λ = 0 results in an **Ordinary Least Squares (OLS) solution**, where all coefficients are nonzero.

#### **5. Highly Correlated Features (test_highly_correlated_features)**
- Runs the model on data with extreme correlation (0.99).
- Ensures that sparsity is maintained.

### **Additional Debugging and Visualization**
- **Plots** the evolution of λ (`plot_lambda_history()`) and θ (`plot_theta_history()`) over iterations.
- **Prints nonzero coefficients** to verify sparsity.

---

## 2.3. What parameters have you exposed to users of your implementation in order to tune performance?

The model exposes several hyperparameters that allow users to fine-tune performance:

| Parameter        | Description |
|-----------------|-------------|
| `lambda_init`   | Initial regularization parameter $\lambda$ (controls sparsity). |
| `eta`           | Learning rate for updating $\lambda$ dynamically. |
| `max_iter`      | Maximum number of iterations. |
| `tol`           | Tolerance for stopping criteria (affects convergence speed). |

These parameters allow the user to **control sparsity, convergence speed, and solution stability**.

---

## 2.4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes, the model has certain limitations:

### **Potential Issues:**
1. **Ill-conditioned feature matrices**
   - If $X^T X$ is **not invertible**, the `_homotopy_update` method attempts to use the **pseudoinverse (pinv)**.
   - This can introduce numerical instability.

2. **Extremely low λ values**
   - If λ is **too small**, the model behaves like **OLS**, losing the sparsity advantage.

3. **Highly correlated features**
   - Although the Homotopy Method is designed to handle collinearity, in extreme cases, selecting the active set becomes **less stable**.

4. **Dynamic λ update might be unstable**
   - The exponential update rule:
```math
     \lambda_{n+1} = \lambda_n \times \exp(-\eta \times \text{error})
```
     can lead to rapid changes in λ, making the solution oscillate.

### **Possible Workarounds:**
- **Regularization adjustment**: Introduce a **lower bound** for λ to avoid instability.
- **Better feature selection**: Use **Principal Component Analysis (PCA)** to reduce collinearity before applying LASSO.
- **Adaptive λ update**: Modify λ update rule to **prevent large fluctuations**.

---

# 3. Code Execution.

## 3.1. Set up the Environment

#### Step 1: Create and activate the virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install project dependencies
All required packages are listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## 3.2. Running the LASSO Homotopy Model

The model can be runed on a dataset using the `test_LassoHomotopy.py` script inside the `tests/` directory. This script contains multiple test functions that:
- Train the model on synthetic or CSV datasets.
- Display results and plots.
- Evaluate performance with metrics like Mean Squared Error (MSE).
- Test sparsity behavior under different configurations.

#### Step 1: Navigate to the tests directory
```bash
cd LassoHomotopy/tests
```
#### Step 2: Run all tests using PyTest
To automatically run the tests we use pytest:
```bash
pytest
```
If you want to see the prints of the test run the following
```bash
pytest -s
```

Each test will produce visualizations:
- Bar plots of learned coefficients.
- Line plots showing how λ and θ evolve over time.

---

## **Final Thoughts**
**LassoHomotopyModel** is well-implemented and efficiently computes sparse solutions for LASSO regression. It has been rigorously tested with synthetic and real-world data, ensuring its reliability. However, handling **ill-conditioned data** and tuning **λ updates** could remain areas for improvement.

