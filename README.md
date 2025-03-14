# PAPER SUMMARY: LASSO Regression using the Homotopy Method

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

# Answer the following questions.

## * What does the model you have implemented do and when should it be used?
## * How did you test your model to determine if it is working reasonably correctly?
## * What parameters have you exposed to users of your implementation in order to tune performance? 
## * Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
