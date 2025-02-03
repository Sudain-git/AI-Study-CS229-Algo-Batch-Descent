import numpy as np
import matplotlib.pyplot as plt

class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000, debug=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.debug = debug
        
    def initialize_parameters(self, n_features):
        """Initialize weights and bias to zeros"""
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    def visualize_matrix_multiplication(self, A, B, operation_name):
        """Visualize matrix multiplication steps"""
        print(f"\n{'-'*50}")
        print(f"{operation_name}:")
        print(f"Matrix A ({A.shape}):")
        print(A)
        print(f"\nMatrix B ({B.shape}):")
        print(B)
        
        result = np.dot(A, B)
        print(f"\nResult ({result.shape}):")
        print(result)
        
        if min(A.shape) <= 10 and min(B.shape) <= 10:
            print("\nDetailed multiplication steps:")
            for i in range(A.shape[0]):
                for j in range(B.shape[1] if len(B.shape) > 1 else 1):
                    step = []
                    for k in range(A.shape[1]):
                        step.append(f"({A[i][k]} × {B[k] if len(B.shape) == 1 else B[k][j]})")
                    print(f"Result[{i}] = {' + '.join(step)} = {result[i]}")
        
        print(f"{'-'*50}\n")
        input("Press Enter to continue...")
        return result
        
    def compute_cost(self, X, y):
        """Compute mean squared error"""
        m = len(y)
        if self.debug:
            predictions = self.visualize_matrix_multiplication(
                X, self.weights, "Computing predictions (X · weights)")
            predictions = predictions + self.bias
            print(f"\nAdding bias {self.bias} to predictions:")
            print(predictions)
            print(f"\nTrue values (y):")
            print(y)
            
            errors = predictions - y
            print(f"\nErrors (predictions - y):")
            print(errors)
            
            squared_errors = errors ** 2
            print(f"\nSquared errors:")
            print(squared_errors)
            
            cost = (1/(2*m)) * np.sum(squared_errors)
            print(f"\nFinal cost (mean squared error):")
            print(cost)
            input("Press Enter to continue...")
        else:
            predictions = np.dot(X, self.weights) + self.bias
            cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def compute_gradients(self, X, y):
        """Compute gradients for weights and bias"""
        m = len(y)
        if self.debug:
            print("\nComputing gradients:")
            predictions = self.visualize_matrix_multiplication(
                X, self.weights, "Computing predictions (X · weights)")
            predictions = predictions + self.bias
            
            errors = predictions - y
            print(f"\nComputing errors (predictions - y):")
            print(errors)
            
            # Gradient for weights
            dw = self.visualize_matrix_multiplication(
                X.T, errors, "Computing weight gradients (X.T · errors)")
            dw = dw / m
            
            # Gradient for bias
            db = np.sum(errors) / m
            print(f"\nComputing bias gradient (mean of errors):")
            print(db)
        else:
            predictions = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
        return dw, db
    
    def fit(self, X, y):
        """Train the model using batch gradient descent"""
        self.initialize_parameters(X.shape[1])
        
        for i in range(self.n_iterations):
            if self.debug and i < 3:  # Only show first 3 iterations in debug mode
                print(f"\nIteration {i+1}:")
                dw, db = self.compute_gradients(X, y)
                
                print(f"\nCurrent weights: {self.weights}")
                print(f"Weight update: -{self.learning_rate} × {dw}")
                self.weights = self.weights - self.learning_rate * dw
                print(f"New weights: {self.weights}")
                
                print(f"\nCurrent bias: {self.bias}")
                print(f"Bias update: -{self.learning_rate} × {db}")
                self.bias = self.bias - self.learning_rate * db
                print(f"New bias: {self.bias}")
                
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                
                if i < 2:  # Ask to continue except for last debug iteration
                    input("\nPress Enter to continue to next iteration...")
            else:
                dw, db = self.compute_gradients(X, y)
                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
            
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias

# Example usage with small dataset
if __name__ == "__main__":
    # Generate small sample data
    np.random.seed(42)
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ])
    y = np.array([2, 4, 6, 8, 10])
    
    # Create and train model with debug mode
    model = BatchGradientDescent(learning_rate=0.01, n_iterations=1000, debug=True)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    print("\nFinal predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, True value: {y[i]}, Predicted: {predictions[i]:.2f}")