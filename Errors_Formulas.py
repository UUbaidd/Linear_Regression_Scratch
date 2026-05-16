
def R2_scratch(y_true, y_pred):
    y_mean = np.mean(y_true)
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - y_mean)**2)
    return 1 - (rss / tss)

def Adjust_R2(n, k, r2_val):
    return 1 - ((1 - r2_val) * (n - 1) / (n - k - 1))

def MAE_scratch(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
  
def MSE_scratch(y_true, y_pred):
    
    return np.mean((y_true - y_pred) ** 2)
