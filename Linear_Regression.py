import numpy as np

class LinearRegressionGD:
    def __init__(self, eta=0.1, n_iter=1000, loss_type='huber', delta=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.loss_type = loss_type 
        self.delta = delta         
        self.w_ = None
        self.b_ = None
        self.cost_history = []


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features)
        self.b_ = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.w_)  + self.b_
            error = y_pred - y 

            if self.loss_type == 'mse':
                dw = (2 / n_samples) * np.dot(X.T, error)
                db = (2 / n_samples) * np.sum(error)
                cost = np.mean(error**2)
            
            elif self.loss_type == 'huber':
                # Gradient of Huber Loss
                huber_grad = np.where(np.abs(error) <= self.delta, 
                                      error, 
                                      self.delta * np.sign(error))
                dw = (1 / n_samples) * np.dot(X.T, huber_grad)
                db = (1 / n_samples) * np.sum(huber_grad)
                
                # Huber Cost Calculation
                cost = np.mean(np.where(np.abs(error) <= self.delta,
                                        0.5 * error**2,
                                        self.delta * (np.abs(error) - 0.5 * self.delta)))

            self.w_ -= self.eta * dw
            self.b_ -= self.eta * db
            self.cost_history.append(cost)
        return self
