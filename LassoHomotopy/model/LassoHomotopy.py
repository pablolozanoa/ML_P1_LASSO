import numpy as np
import matplotlib.pyplot as plt

class LassoHomotopyModel:
    def __init__(self, lambda_init=0.1, eta=0.01, max_iter=100, tol=1e-6):
        self.lambda_reg = lambda_init  # Regularization parameter
        self.eta = eta  # Learning rate for lambda update
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Convergence tolerance
        self.theta = None  # Model coefficients
        self.active_set = set()  # Set of active features
        self.lambda_history = []  # Store lambda updates
        self.theta_history = []  # Store theta updates

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n, m = X.shape
        self.theta = np.zeros(m)  # Initialize coefficients
        self.active_set = set()
        
        for _ in range(self.max_iter):
            gradient = X.T @ (X @ self.theta - y)  # Compute gradient
            idx = np.argmax(np.abs(gradient))  # Select the most relevant variable
            
            if idx not in self.active_set:
                self.active_set.add(idx)
            
            active_list = list(self.active_set)
            X_active = X[:, active_list]
            
            if len(active_list) > 0:
                theta_active_prev = self.theta[active_list].copy()
                
                # Homotopy update using rank-one updates
                theta_active = self._homotopy_update(X_active, y, theta_active_prev)
                self.theta[active_list] = theta_active
                
                # More stable handling of coefficient removal
                transition_points = np.where(np.abs(theta_active) < self.tol)[0]
                for t in transition_points:
                    if active_list[t] in self.active_set:
                        self.active_set.remove(active_list[t])
            
            # Store updates for visualization
            self.lambda_history.append(self.lambda_reg)
            self.theta_history.append(self.theta.copy())
            
            # Update regularization parameter dynamically
            prediction_error = np.mean((X @ self.theta - y) ** 2)
            self.lambda_reg *= np.exp(-self.eta * prediction_error)
            
            # Check for convergence
            if np.linalg.norm(gradient, ord=np.inf) < self.tol:
                break
        
        return LassoHomotopyResults(self.theta)
    
    def _homotopy_update(self, X_active, y, theta_active_prev):
        """
        Implements a more precise homotopy update.
        """
        try:
            inv_X = np.linalg.inv(X_active.T @ X_active)
        except np.linalg.LinAlgError:
            inv_X = np.linalg.pinv(X_active.T @ X_active)  # Use pseudoinverse if not invertible
        
        return inv_X @ (X_active.T @ y - self.lambda_reg * np.sign(theta_active_prev))

    def plot_lambda_history(self):
        """ Plot the evolution of lambda_reg over iterations """
        plt.figure(figsize=(8, 5))
        plt.plot(self.lambda_history, label='Lambda Value', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Lambda Value')
        plt.title('Evolution of Regularization Parameter (Lambda)')
        plt.legend()
        plt.show()
    
    def plot_theta_history(self):
        """ Plot the evolution of coefficients over iterations """
        plt.figure(figsize=(10, 6))
        for i in range(len(self.theta_history[0])):
            plt.plot([theta[i] for theta in self.theta_history], label=f'Feature {i}')
        plt.xlabel('Iteration')
        plt.ylabel('Coefficient Value')
        plt.title('Evolution of Coefficients Over Iterations')
        plt.legend()
        plt.show()

class LassoHomotopyResults:
    def __init__(self, theta):
        self.theta = theta
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.theta