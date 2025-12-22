import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.generation import MaxPosteriorSampling
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan


ALLOWED_KERNELS = {
    "matern_2.5": {"class": MaternKernel, "kwargs": {"nu": 2.5}},
    "matern_1.5": {"class": MaternKernel, "kwargs": {"nu": 1.5}},
    "rbf":        {"class": RBFKernel,    "kwargs": {}},
}

ALLOWED_NOISE_PRIORS = {
    "fixed_low":  {"min_noise": 1e-4}, 
    "learnable":  {"min_noise": 1e-5}, 
    "high_noise": {"min_noise": 1e-2}, 
}

def build_covar_module(kernel_key: str, input_dim: int) -> ScaleKernel:
    """Factory for Kernel selection."""
    if kernel_key not in ALLOWED_KERNELS:
        kernel_key = "matern_2.5"
    
    config = ALLOWED_KERNELS[kernel_key]
    base_kernel = config["class"](ard_num_dims=input_dim, **config["kwargs"])
    return ScaleKernel(base_kernel)

def build_likelihood(noise_key: str) -> GaussianLikelihood:
    """Factory for Likelihood/Noise selection."""
    if noise_key not in ALLOWED_NOISE_PRIORS:
        noise_key = "fixed_low"
        
    config = ALLOWED_NOISE_PRIORS[noise_key]
    noise_constraint = GreaterThan(config["min_noise"])
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
    
    # Initialize slightly above min to aid convergence
    likelihood.noise = torch.tensor(config["min_noise"] * 2.0)
    return likelihood


class SingleObjectiveOptimizer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None
        self.y_train = None
        self.bounds = None
        self.input_dim = 0
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: List[Tuple[float, float]], 
            model_config: Dict[str, str], feature_names: List[str] = None):
        """Fits the SingleTaskGP."""
        self.X_train = torch.tensor(X, dtype=torch.double, device=self.device)
        self.y_train = torch.tensor(y, dtype=torch.double, device=self.device)
        if self.y_train.ndim == 1: self.y_train = self.y_train.unsqueeze(-1)
        
        self.input_dim = self.X_train.shape[-1]
        self.bounds = torch.tensor(bounds, dtype=torch.double, device=self.device).T
        self.feature_names = feature_names or [f"x{i}" for i in range(self.input_dim)]

        kernel_choice = model_config.get("kernel", "matern_2.5")
        noise_choice = model_config.get("noise", "fixed_low")
        
        covar = build_covar_module(kernel_choice, self.input_dim)
        likelihood = build_likelihood(noise_choice)

        self.model = SingleTaskGP(
            self.X_train, 
            self.y_train,
            covar_module=covar,
            likelihood=likelihood,
            input_transform=Normalize(d=self.input_dim),
            outcome_transform=Standardize(m=1)
        )
        
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def recommend(self, n_candidates: int = 1, strategy: str = 'log_ei', params: Dict[str, float] = None) -> np.ndarray:
        """
        Generates n_candidates. 
        Supports 'thompson' for high-throughput batches, and 'ucb'/'log_ei' for precision.
        """
        if self.model is None: raise RuntimeError("Call fit() first.")
        params = params or {}

        # --- 1. Thompson Sampling (High Throughput / Diversity) ---
        if strategy == 'thompson':
            n_pool = max(2000, 500 * n_candidates)
            X_cand = draw_sobol_samples(bounds=self.bounds, n=n_pool, q=1).squeeze(1)
            
            thompson_sampler = MaxPosteriorSampling(model=self.model, replacement=False)
            candidates = thompson_sampler(X_cand, num_samples=n_candidates)
            return candidates.detach().cpu().numpy()

        # --- 2. Acquisition Functions ---
        if strategy == 'ucb':
            # Uses beta parameter from Agent
            beta = params.get('beta', 2.0)
            acq_func = qUpperConfidenceBound(model=self.model, beta=beta)
            
        elif strategy == 'max_variance':
            acq_func = qUpperConfidenceBound(model=self.model, beta=1000.0)
            
        else: # Default: 'log_ei'
            best_f = self.y_train.max()
            acq_func = LogExpectedImprovement(model=self.model, best_f=best_f)

        # --- 3. Optimization (Greedy Batch) ---
        is_large_batch = n_candidates > 10
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_candidates,
            num_restarts=2 if is_large_batch else 10,
            raw_samples=128 if is_large_batch else 512,
            sequential=True # Optimizes one point at a time conditioning on previous
        )
        return candidates.detach().cpu().numpy()

    def _compute_sensitivity(self, n_samples=2048) -> Tuple[List[str], List[float]]:
        """Helper: Calculates First-Order Sobol Indices for diagnostics."""
        if self.input_dim == 1: return [self.feature_names[0]], [1.0]

        # 1. Sobol Sampling [0, 1]
        bounds = torch.stack([torch.zeros(self.input_dim), torch.ones(self.input_dim)]).to(self.device)
        X_sobol = draw_sobol_samples(bounds=bounds, n=n_samples * 2, q=1).squeeze(1)
        A, B = X_sobol[:n_samples], X_sobol[n_samples:]

        # 2. Predict Mean
        def predict(X):
            with torch.no_grad(): return self.model.posterior(X).mean.flatten()

        f_A, f_B = predict(A), predict(B)
        var_Y = torch.var(torch.cat([f_A, f_B])) + 1e-9

        # 3. Compute Indices
        indices = []
        for i in range(self.input_dim):
            AB_i = A.clone(); AB_i[:, i] = B[:, i]
            numerator = torch.mean(f_B * (predict(AB_i) - f_A))
            indices.append(max(0.0, (numerator / var_Y).item()))
        return self.feature_names, indices

    def generate_diagnostics(self, candidate_x: np.ndarray, history_y: List[float], save_path: str):
        """Generates 4-Panel Dashboard: Calibration, Trend, Slice, Sensitivity."""
        # Only plot the FIRST candidate to keep visualization clean
        x_plot = candidate_x[0:1]

        y_np = self.y_train.cpu().numpy().flatten()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # --- 1. Calibration (Top Left) ---
        ax_cal = axes[0, 0]
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(self.X_train)
            pred_mean = posterior.mean.cpu().numpy().flatten()
            pred_std = posterior.variance.sqrt().cpu().numpy().flatten()
        ax_cal.errorbar(y_np, pred_mean, yerr=pred_std, fmt='o', alpha=0.5, label='Predictions')
        min_v, max_v = y_np.min(), y_np.max()
        buff = (max_v - min_v) * 0.1 if max_v != min_v else 0.1
        ax_cal.plot([min_v-buff, max_v+buff], [min_v-buff, max_v+buff], 'r--', label='Ideal')
        ax_cal.set_title("1. Model Accuracy"); ax_cal.set_xlabel("Observed"); ax_cal.set_ylabel("Predicted"); ax_cal.legend()

        # --- 2. Trend (Top Right) ---
        ax_trend = axes[0, 1]
        steps = np.arange(1, len(history_y) + 1)
        ax_trend.plot(steps, history_y, 'ko-', alpha=0.3)
        ax_trend.plot(steps, np.maximum.accumulate(history_y), 'g-', linewidth=2, label='Best Found')
        ax_trend.set_title("2. Optimization Trend")

        # --- 3. Sensitivity (Bottom Right) ---
        ax_sens = axes[1, 1]
        top_dim_idx = 0
        try:
            names, scores = self._compute_sensitivity()
            sorted_idx = np.argsort(scores)[::-1] # Descending
            top_dim_idx = sorted_idx[0]
            
            y_pos = np.arange(len(names))
            ax_sens.barh(y_pos, [scores[i] for i in sorted_idx], align='center', color='skyblue')
            ax_sens.set_yticks(y_pos)
            ax_sens.set_yticklabels([names[i] for i in sorted_idx])
            ax_sens.invert_yaxis()
            ax_sens.set_xlabel('Sobol Index (Impact on Mean)')
            ax_sens.set_title("4. Parameter Importance")
        except Exception as e:
            ax_sens.text(0.5, 0.5, f"Analysis Error: {str(e)}", ha='center')

        # --- 4. Slice (Bottom Left) ---
        # Slices along the most important dimension found above
        ax_slice = axes[1, 0]
        dim_name = self.feature_names[top_dim_idx]
        b_min = self.bounds[0, top_dim_idx].item()
        b_max = self.bounds[1, top_dim_idx].item()
        
        x_sweep = np.linspace(b_min, b_max, 100)
        X_slice = np.tile(x_plot, (100, 1))
        X_slice[:, top_dim_idx] = x_sweep
        
        with torch.no_grad():
            post_slice = self.model.posterior(torch.tensor(X_slice, dtype=torch.double, device=self.device))
            mu = post_slice.mean.cpu().numpy().flatten()
            sigma = post_slice.variance.sqrt().cpu().numpy().flatten()

        ax_slice.plot(x_sweep, mu, 'b-', label='Mean')
        ax_slice.fill_between(x_sweep, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='b')
        ax_slice.axvline(x_plot[0, top_dim_idx], color='green', linestyle='--', label='Next Point')
        ax_slice.set_title(f"3. Slice along '{dim_name}' (Top Factor)")
        ax_slice.legend()

        plt.tight_layout(); plt.savefig(save_path); plt.close()


class MultiObjectiveOptimizer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None
        self.y_train = None
        self.bounds = None
        self.input_dim = 0
        self.output_dim = 0
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: List[Tuple[float, float]], 
            model_config: Dict[str, str], feature_names: List[str] = None):
        self.X_train = torch.tensor(X, dtype=torch.double, device=self.device)
        self.y_train = torch.tensor(y, dtype=torch.double, device=self.device)
        self.input_dim = self.X_train.shape[-1]
        self.output_dim = self.y_train.shape[-1]
        self.bounds = torch.tensor(bounds, dtype=torch.double, device=self.device).T
        self.feature_names = feature_names or [f"x{i}" for i in range(self.input_dim)]
        # Independent GPs
        models = []
        for i in range(self.output_dim):
            kernel_choice = model_config.get("kernel", "matern_2.5")
            noise_choice = model_config.get("noise", "fixed_low")
            
            models.append(
                SingleTaskGP(
                    self.X_train, self.y_train[:, i : i + 1],
                    covar_module=build_covar_module(kernel_choice, self.input_dim),
                    likelihood=build_likelihood(noise_choice),
                    input_transform=Normalize(d=self.input_dim),
                    outcome_transform=Standardize(m=1)
                )
            )
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def recommend(self, n_candidates: int = 1, strategy: str = 'pareto', params: Dict[str, Any] = None) -> np.ndarray:
        if self.model is None: raise RuntimeError("Call fit() first.")
        params = params or {}

        if strategy == 'weighted':
            # Scalarized UCB
            weights = params.get('weights', [1.0]*self.output_dim)
            beta = params.get('beta', 0.1) # Handles beta for Weighted UCB
            weights_t = torch.tensor(weights, device=self.device)
            objective = LinearMCObjective(weights=weights_t)
            acq_func = qUpperConfidenceBound(model=self.model, beta=beta, objective=objective)
            
        elif strategy == 'max_variance':
            weights = params.get('weights', [1.0]*self.output_dim)
            weights_t = torch.tensor(weights, device=self.device)
            objective = LinearMCObjective(weights=weights_t)
            acq_func = qUpperConfidenceBound(model=self.model, beta=1000.0, objective=objective)
            
        else: # Default: 'pareto'
            ref_point = self.y_train.min(dim=0)[0] - 0.1 * torch.abs(self.y_train.min(dim=0)[0])
            acq_func = qLogNoisyExpectedHypervolumeImprovement(
                model=self.model, 
                X_baseline=self.X_train, 
                prune_baseline=True, 
                ref_point=ref_point
            )

        # Batch Optimization
        is_large = n_candidates > 10
        candidates, _ = optimize_acqf(
            acq_function=acq_func, bounds=self.bounds, q=n_candidates,
            num_restarts=2 if is_large else 10, 
            raw_samples=128 if is_large else 256,
            sequential=True
        )
        return candidates.detach().cpu().numpy()

    def generate_diagnostics(self, save_path: str):
        """
        Generates Pairwise Pareto Scatter Plots for ANY N >= 2.
        - N=2: Generates 1 plot.
        - N=3: Generates 3 plots.
        - N=4: Generates 6 plots.
        """
        import itertools
        import math
        
        y_np = self.y_train.cpu().numpy()
        
        # 1. Non-dominated sort (Identify Pareto Efficient points)
        is_efficient = np.ones(y_np.shape[0], dtype=bool)
        for i, c in enumerate(y_np):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(y_np[is_efficient] > c, axis=1) | (y_np[is_efficient] == c).all(axis=1)

        # 2. Dynamic Plotting Logic
        # Create all unique pairs: (0,1), (0,2), (1,2)...
        pairs = list(itertools.combinations(range(self.output_dim), 2))
        n_plots = len(pairs)
        
        # Determine Grid Dimensions
        if n_plots == 1:
            rows, cols = 1, 1
            figsize = (8, 6)
        else:
            cols = min(3, n_plots)
            rows = math.ceil(n_plots / cols)
            figsize = (5 * cols, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Standardize axes object to always be iterable
        if n_plots == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
        
        # 3. Plot Loop
        for idx, (dim_x, dim_y) in enumerate(pairs):
            ax = axes_list[idx]
            
            # Plot Dominated (Gray)
            ax.scatter(y_np[~is_efficient][:, dim_x], y_np[~is_efficient][:, dim_y], 
                       c='gray', alpha=0.3, label='Dominated' if idx == 0 else "")
            
            # Plot Efficient (Red)
            ax.scatter(y_np[is_efficient][:, dim_x], y_np[is_efficient][:, dim_y], 
                       c='red', edgecolors='darkred', s=40, label='Pareto' if idx == 0 else "")
            
            # Labels: Use column names if available, else generic "Obj 1"
            name_x = self.feature_names[dim_x] if self.feature_names and dim_x < len(self.feature_names) else f"Obj {dim_x+1}"
            name_y = self.feature_names[dim_y] if self.feature_names and dim_y < len(self.feature_names) else f"Obj {dim_y+1}"
            
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)
            ax.set_title(f"{name_x} vs {name_y}")
            ax.grid(True, linestyle='--', alpha=0.5)

            if idx == 0: ax.legend()

        # Hide empty subplots in the grid
        for i in range(n_plots, len(axes_list)):
            axes_list[i].axis('off')

        plt.suptitle(f"Multi-Objective Diagnostics ({self.output_dim} Objectives)", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def get_optimizer(is_moo: bool, device: str = "cpu"):
    return MultiObjectiveOptimizer(device) if is_moo else SingleObjectiveOptimizer(device)