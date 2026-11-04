"""Module for continuous-time Bayesian inference and exact gradient descent MSE predictions."""

import torch
from typing import Callable, Optional, Union, Sequence, NamedTuple, Tuple
from torchtangents._src.utils.kernel import Kernel


class PredictOut(NamedTuple):
    mean: torch.Tensor
    covariance: Optional[torch.Tensor] = None


def _get_diag_reg(mat: torch.Tensor, diag_reg: float, diag_reg_absolute_scale: bool) -> float:
    if diag_reg == 0.0:
        return 0.0
    if diag_reg_absolute_scale:
        return diag_reg
    else:
        # Default relative regularization to the trace
        return diag_reg * (torch.trace(mat) / mat.shape[0]).item()


def gp_inference(
    k_train_train: Union[torch.Tensor, Kernel],
    y_train: torch.Tensor,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False
) -> Callable:
    """
    Computes the exact posterior mean and covariance for Bayesian inference.

    Args:
        k_train_train: The train-train kernel evaluation matrix or Kernel object.
        y_train: The training labels of shape (N_train, C).
        diag_reg: Ridge regularization (added to diagonal)
        diag_reg_absolute_scale: whether diag_reg is an absolute scale or relative.

    Returns:
        A function `predict_fn(get, k_test_train, k_test_test)` that returns 
        the evaluated exact posterior inference predictions dynamically while 
        caching the expensive Cholesky evaluation of k_train_train.
    """
    # Extract matrices
    nngp_train = k_train_train.nngp if hasattr(k_train_train, 'nngp') else (None if hasattr(k_train_train, 'ntk') else k_train_train)
    ntk_train = k_train_train.ntk if hasattr(k_train_train, 'ntk') else (None if hasattr(k_train_train, 'nngp') else k_train_train)

    # Pre-factorize for NNGP inference
    L_nngp, alpha_nngp = None, None
    if nngp_train is not None:
        reg = _get_diag_reg(nngp_train, diag_reg, diag_reg_absolute_scale)
        nngp_train_reg = nngp_train + reg * torch.eye(nngp_train.shape[0], device=nngp_train.device, dtype=nngp_train.dtype)
        L_nngp = torch.linalg.cholesky(nngp_train_reg)
        alpha_nngp = torch.cholesky_solve(y_train, L_nngp)

    # Pre-factorize for NTK inference (GP trained endlessly on MSE)
    L_ntk, alpha_ntk = None, None
    if ntk_train is not None:
        reg = _get_diag_reg(ntk_train, diag_reg, diag_reg_absolute_scale)
        ntk_train_reg = ntk_train + reg * torch.eye(ntk_train.shape[0], device=ntk_train.device, dtype=ntk_train.dtype)
        L_ntk = torch.linalg.cholesky(ntk_train_reg)
        alpha_ntk = torch.cholesky_solve(y_train, L_ntk)

    def predict_fn(
        get: str,
        k_test_train: Union[torch.Tensor, Kernel],
        k_test_test: Optional[Union[torch.Tensor, Kernel]] = None
    ) -> PredictOut:
        if get not in ('nngp', 'ntk'):
            raise ValueError(f"`get` must be either 'nngp' or 'ntk'. Got {get}.")

        # Retrieve specific matrices
        test_train = k_test_train.nngp if hasattr(k_test_train, 'nngp') and get == 'nngp' else (
                     k_test_train.ntk if hasattr(k_test_train, 'ntk') and get == 'ntk' else k_test_train)
        
        test_test = None
        test_test_nngp = k_test_test.nngp if hasattr(k_test_test, 'nngp') else k_test_test
        test_test_ntk = k_test_test.ntk if hasattr(k_test_test, 'ntk') else k_test_test
        test_test = test_test_nngp if get == 'nngp' else test_test_ntk

        # Retrieve factored weights based on `get` argument
        L, alpha = (L_nngp, alpha_nngp) if get == 'nngp' else (L_ntk, alpha_ntk)

        if L is None or alpha is None:
            raise ValueError(f"Could not compute `gp_inference` because {get} matrix was not supplied in `k_train_train`.")

        mean = test_train @ alpha
        
        covariance = None
        if test_test is not None:
            if get == 'nngp':
                v = torch.linalg.solve_triangular(L, test_train.T, upper=False)
                covariance = test_test - v.T @ v
            else:
                # NTK ensemble posterior exact evaluation.
                # Covariance = NNGP_test_test + K_te_tr @ K_tr_tr^-1 @ NNGP_tr_tr @ K_tr_tr^-1 @ K_tr_te - 2 * K_te_tr @ K_tr_tr^-1 @ NNGP_tr_te
                K_te_tr_NTK = test_train
                # A = K_tr_tr^-1 @ K_tr_te   ->    L @ L^T @ A = K_tr_te -> A = L^-T @ L^-1 @ K_tr_te
                v = torch.linalg.solve_triangular(L, K_te_tr_NTK.T, upper=False)
                K_inv_K_tr_te = torch.linalg.solve_triangular(L.T, v, upper=True)
                
                nngp_test_train = k_test_train.nngp if hasattr(k_test_train, 'nngp') else k_test_train
                covariance = test_test_nngp + K_inv_K_tr_te.T @ nngp_train @ K_inv_K_tr_te - K_inv_K_tr_te.T @ nngp_test_train.T - nngp_test_train @ K_inv_K_tr_te

        return PredictOut(mean=mean, covariance=covariance)

    return predict_fn


def gradient_descent_mse(
    k_train_train: Union[torch.Tensor, Kernel],
    y_train: torch.Tensor,
    learning_rate: float = 1.0,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False
) -> Callable:
    """
    Predicts the outcome of function-space continuous-time gradient descent training on MSE.
    
    Solves in closed form for the continuous-time version of gradient descent using the 
    eigendecomposition, cached natively for infinite time evaluations.

    Args:
        k_train_train: The train-train kernel evaluation matrix or Kernel object.
        y_train: The targets for the training data (N_train, C).
        learning_rate: step size, scaling the time `t` evaluating continuous equations.
        diag_reg: Ridge regularization value.
        
    Returns:
        A function `predict_fn(t, fx_train_0, fx_test_0, k_test_train)` that
        returns output (train_predictions, test_predictions) at continuous time `t`.
    """
    
    K_train = k_train_train.ntk if hasattr(k_train_train, 'ntk') else k_train_train
    if K_train is None:
        raise ValueError("Provided k_train_train must contain NTK to predict gradient descent.")

    reg = _get_diag_reg(K_train, diag_reg, diag_reg_absolute_scale)
    K_reg = K_train + reg * torch.eye(K_train.shape[0], device=K_train.device, dtype=K_train.dtype)

    # Cache for factorization metrics preventing double compilation overhead
    cache = {}

    def predict_fn(
        t: Union[float, torch.Tensor, None],
        fx_train_0: torch.Tensor,
        fx_test_0: Optional[torch.Tensor] = None,
        k_test_train: Optional[Union[torch.Tensor, Kernel]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate function mappings perfectly accurately at temporal gradient step `t`.
        If t == None, evaluates at strictly t = infinity.
        """
        if t is None:
            # Infinite time exclusively relies on Cholesky to avoid numeric eigendecomposition limits
            if 'cho' not in cache:
                cache['cho'] = torch.linalg.cholesky(K_reg)
            L = cache['cho']
            
            # Infinite time strictly implies training residual converges identically to y_train.
            fx_train_t = y_train.clone().to(K_train.dtype)
            
            if fx_test_0 is None or k_test_train is None:
                return fx_train_t
                
            k_test_train_tensor = k_test_train.ntk if hasattr(k_test_train, 'ntk') else k_test_train
            
            # Predict test with Cholesky solve: fx_test_0 + K_te_tr @ K_tr_tr^-1 @ (y_train - fx_train_0)
            res = y_train - fx_train_0
            dfx_test = k_test_train_tensor @ torch.cholesky_solve(res, L)
            fx_test_t = fx_test_0 + dfx_test
            
            return fx_train_t, fx_test_t
            
        else:
            # Finite time ODE projection via Eigenbasis spectral map
            if 'eigen' not in cache:
                cache['eigen'] = torch.linalg.eigh(K_reg)
            eigenvals, Q = cache['eigen']
            
            time_decay = 1.0 - torch.exp(-learning_rate * t * torch.clamp(eigenvals, min=0.0) / y_train.numel())
            
            # (I - exp(-eta * K * t)) @ y
            train_residual = y_train - fx_train_0
            
            Q_T_decayed = time_decay.unsqueeze(-1) * (Q.T @ train_residual)   # shape (N, C)
            
            # Exact train prediction offset mapping
            pred_offset_train = Q @ Q_T_decayed
            fx_train_t = fx_train_0 + pred_offset_train
            
            if fx_test_0 is None or k_test_train is None:
                return fx_train_t
                
            k_test_train_tensor = k_test_train.ntk if hasattr(k_test_train, 'ntk') else k_test_train
            
            # Test predictions
            # Where K^-1 relies on eigendecomposition = Q @ Lambda^-1 @ Q.T
            lambda_inv = 1.0 / torch.clamp(torch.abs(eigenvals), min=1e-12)
            
            inv_decayed = lambda_inv.unsqueeze(-1) * Q_T_decayed
            Q_inv_decayed = Q @ inv_decayed
            
            pred_offset_test = k_test_train_tensor @ Q_inv_decayed
            fx_test_t = fx_test_0 + pred_offset_test
            
            return fx_train_t, fx_test_t

    return predict_fn
