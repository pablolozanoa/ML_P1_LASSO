# Project 1 

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an altexrnative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

# LASSO Regression using the Homotopy Method

## 1. Introduction to LASSO
LASSO (Least Absolute Shrinkage and Selection Operator) is a regression technique that introduces an ℓ₁ penalty to induce **sparsity** in the coefficient vector θ.

### Objective
To fit a linear model by minimizing the squared error with a penalty:
```math
\min_{\theta} \frac{1}{2} \sum_{i=1}^{n} (x_i^T \theta - y_i)^2 + \mu_n \left\|\theta \right\|_1
```
Where $\mu_n$ is the regularization parameter. The solution θ tends to be **sparse**, meaning many coefficients are zero, which helps with feature selection.

---

## 2. Why Use the Homotopy Method?
Traditional solvers like *Coordinate Descent* or *Least Angle Regression (LARS)* can be computationally expensive.

### Advantages of the Homotopy Method
- **Leverages previous calculations**: If we already have the solution for a dataset, we can update it **efficiently** when adding new data points.
- **Avoids recomputation from scratch**: It follows a trajectory of solutions as new information is introduced.

---

## 3. Optimality Conditions for LASSO
Since the objective function is convex but non-differentiable (due to the ℓ₁ norm), the optimal solution is characterized by the **subdifferential** of $\left\|\theta \right\|_1$:
```math
X^T (X \theta - y) + \mu_n v = 0, \quad v \in \partial \left\|\theta \right\|_1
```
Where $v$ is a vector of subgradients.

By defining the **active set** (A), which consists of the variables with nonzero coefficients in $\theta$, we can rewrite the solution in a closed form for the active coefficients.

---

## 4. Proposed Homotopy Algorithm (RecLasso)
The algorithm has **two main steps** when a new data point $(y_{n+1}, x_{n+1})$ is introduced:

### 4.1 Step 1: Update the Regularization Parameter $\mu_n$
If we want to change $\mu_n$ to a new value $\mu_{n+1}$, we efficiently follow the LASSO solution path.

### 4.2 Step 2: Vary the Parameter \( t \) from 0 to 1
We define the following problem:
```math
\theta(t, \mu) = \arg \min_{\theta} \frac{1}{2} \left\|(X, t x_{n+1}) \theta - (y, t y_{n+1})\right\|\tfrac{2}{2} + \mu \left\|\theta \right\|_1

```
This parameter $t$ allows us to continuously update the solution as the new observation is added.

### 4.3 Computing Transition Points
- As $t$ increases, the solution $\theta(t)$ changes **smoothly** until a change in the active set occurs (a coefficient becomes zero, or a new coefficient becomes active).
- The next transition point is computed, where this change occurs, and the solution is updated.

---

## 5. Applications
### 5.1 Compressive Sensing
- Used when the signal is **sparse** and we have fewer measurements than parameters ($n < m$).
- **RecLasso** is more efficient than *LARS* and *Coordinate Descent* for sequentially updating solutions.

### 5.2 Regularization Parameter $\lambda$ Selection
- A **data-driven method** is proposed to dynamically adjust $\lambda$ using the prediction error of the new observation.
- The update is computed as:
```math
\lambda_{n+1} = \lambda_n \times \exp\left(2n \eta x_{n+1,1}^T (X_1^T X_1)^{-1} v_1 (x_{n+1}^T \theta - y_{n+1})\right)
```

### 6.3 Leave-One-Out Cross-Validation
- The algorithm is adapted to **remove data points** instead of adding them.
- Useful for selecting the optimal $\lambda$ via **cross-validation**.

---

## 7. Results
- **Homotopy is faster and more stable than LARS** in sequential problems.
- **Handles collinear data efficiently.**

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
