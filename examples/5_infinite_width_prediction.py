"""5. Infinite-Width Prediction: Train an infinitely wide network in closed form."""
import torch
import torchtangents as tt

x_train = torch.randn(16, 8)
y_train = torch.sin(x_train[:, :1]).to(torch.float64)
x_test = torch.randn(4, 8)

kernel_fn = tt.stax.Sequential(
    tt.stax.Linear(256),
    tt.stax.ReLU(),
    tt.stax.Linear(1),
)

k_train_train = kernel_fn(x_train, None, get=("nngp", "ntk"))
k_test_train = kernel_fn(x_test, x_train, get=("nngp", "ntk"))

# 1. Exact Bayesian GP / NTK ensemble posterior inference (no training loop!)
gp_predict = tt.predict.gp_inference(k_train_train, y_train, diag_reg=1e-4)
ntk_pred = gp_predict(get="ntk", k_test_train=k_test_train)
nngp_pred = gp_predict(get="nngp", k_test_train=k_test_train)
print(f"Bayesian NTK / NNGP test predictions: {ntk_pred.mean.shape}, {nngp_pred.mean.shape}")

# 2. Continuous-time gradient descent MSE trajectory at finite time t=1.0
gd_predict = tt.predict.gradient_descent_mse(k_train_train, y_train, diag_reg=1e-4)
fx_train_0 = torch.zeros_like(y_train)
fx_test_0 = torch.zeros(len(x_test), 1, dtype=torch.float64)
fx_train_t, fx_test_t = gd_predict(t=1.0, fx_train_0=fx_train_0, fx_test_0=fx_test_0, k_test_train=k_test_train)
print(f"GD (t=1.0) train / test predictions:  {fx_train_t.shape}, {fx_test_t.shape}")
