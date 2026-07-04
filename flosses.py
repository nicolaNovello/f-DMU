import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedChiSquaredLoss(nn.Module):
    """
    Implements the training objective based on the Chi-squared divergence for two
    Gaussian distributions with a shared covariance.
    """
    def __init__(self, sigma: float = 1.0, omega_t: float = 1.0):
        """
        Args:
            sigma (float): The standard deviation of the Gaussian distributions. Must be positive.
            omega_t (float): A weighting factor for the loss.
        """
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        
        self.denominator = sigma**2
        self.omega_t = omega_t

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Chi-squared objective loss.
        
        Args:
            pred (torch.Tensor): The output corresponding to the concept to be forgotten.
            target (torch.Tensor): The output corresponding to the anchor concept. 
            mask (torch.Tensor): A binary mask with the same shape as pred/target to select which
                                 elements contribute to the loss.
        
        Returns:
            torch.Tensor: The final scalar loss value, averaged over the batch.
        """
        # Calculate the squared difference between prediction and a detached target.
        per_element_loss = torch.pow(pred - target.detach(), 2)
        masked_squared_l2_norm = (per_element_loss * mask).sum(dim=tuple(range(1, pred.dim()))) / mask.sum(tuple(range(1, pred.dim())))
        
        # Apply the Chi-squared objective formula per sample.
        loss_per_sample = self.omega_t * torch.exp(masked_squared_l2_norm / self.denominator)
        
        # Average the loss over the batch to get a single scalar value.
        return loss_per_sample.mean()


class MaskedSquaredHellingerLoss(nn.Module):
    def __init__(self, sigma: float=1.0, omega_t: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.neg_denominator = - 8 * sigma**2
        self.omega_t = omega_t
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted distribution from the concept we want to forget.
            target (torch.Tensor): Target distribution using the anchor concept.
            mask (torch.Tensor): Mask with the same shape of pred/target.
        
        Returns:
            torch.Tensor: Loss (scalar).
        """
        # MSE between prediction and target
        per_element_loss = torch.pow(pred - target.detach(), 2)
        masked_squared_l2_norm = (per_element_loss * mask).sum(dim=tuple(range(1, pred.dim()))) / mask.sum(tuple(range(1, pred.dim())))
        
        # Hellinger loss per sample
        loss_per_sample = - self.omega_t * torch.exp( masked_squared_l2_norm / self.neg_denominator )

        # Average over the batch
        return loss_per_sample.mean()


class MaskedMSE(nn.Module):
    def __init__(self, sigma: float = 1.0, omega_t: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.omega_t = omega_t
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted distribution from the concept we want to forget.
            target (torch.Tensor): Target distribution using the anchor concept.
            mask (torch.Tensor): Mask with the same shape of pred/target.

        Returns:
            torch.Tensor: Loss (scalar).
        """
        # MSE between prediction and target
        per_element_loss = torch.pow(pred - target.detach(), 2)

        # Apply the mask and sum over feature dimension 
        masked_squared_l2_norm = (per_element_loss * mask).sum(dim=tuple(range(1, pred.dim()))) / mask.sum(tuple(range(1, pred.dim())))

        # Average over the batch
        return masked_squared_l2_norm.mean()


from typing import Callable, Dict    
    
_DIVERGENCES: Dict[str, Dict[str, Callable]] = {
    'kl': {
        'activation': lambda v: v,
        'conjugate': lambda t, eps: torch.exp(t - 1.0),
    },
    'reverse_kl': {
        'activation': lambda v: -torch.exp(-v),
        'conjugate': lambda t, eps: -1.0 - torch.log(-t + eps),
    },
    'hellinger': {
        'activation': lambda v: 1.0 - torch.exp(-v),
        'conjugate': lambda t, eps: t / (1.0 - t + eps),
    },
    'jensen_shannon': {
        'activation': lambda v: math.log(2.)+F.logsigmoid(v),
        'conjugate': lambda t, eps: -torch.log(2.0 - torch.exp(t) + eps),
    },
    'pearson_chi2': {
        'activation': lambda v: v,
        'conjugate': lambda t, eps: 0.25 * t.pow(2) + t,
    },
    'total_variation': {
        'activation': lambda v: 0.5 * torch.tanh(v),
        'conjugate': lambda t, eps: t,
    },
}



class FDivergenceLoss(nn.Module):
    """
    Calculates the discriminator/critic loss for a given f-divergence,
    with optional support for spatial masking.

    The loss for the discriminator is to maximize:
        V = E_p[T(x)] - E_q[f*(T(x))]
    This is equivalent to minimizing:
        Loss = E_q[f*(T(x))] - E_p[T(x)]

    """
    def __init__(self, divergence_name: str, epsilon: float = 1e-6):
        super().__init__()
        
        divergence_name = divergence_name.lower()
        if divergence_name not in _DIVERGENCES:
            raise ValueError(
                f"Unknown divergence: {divergence_name}. "
                f"Available options are: {list(_DIVERGENCES.keys())}"
            )
            
        self.divergence_name = divergence_name
        self.epsilon = epsilon
        
        self.activation_fn = _DIVERGENCES[divergence_name]['activation']
        self.conjugate_fn = _DIVERGENCES[divergence_name]['conjugate']

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.mean(tensor)
        else:
            while mask.dim() < tensor.dim():
                mask = mask.unsqueeze(1)
            
            return (tensor * mask).sum() / (mask.sum() + self.epsilon)

    def forward(
        self, 
        critic_real_logits: torch.Tensor, 
        critic_fake_logits: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates the discriminator loss.
        ...
        """

        if mask is not None:
            target_h, target_w = critic_real_logits.shape[-2:]
            
            if mask.shape[-2:] != (target_h, target_w):
                if mask.dim() == 3: 
                    mask = mask.unsqueeze(1) 
                
                mask = F.interpolate(mask, size=(target_h, target_w), mode='nearest')

        # Apply the activation function g_f to get T(x)
        t_real = self.activation_fn(critic_real_logits)
        t_fake = self.activation_fn(critic_fake_logits)

        f_star_of_t_fake = self.conjugate_fn(t_fake, self.epsilon)
        
        term1 = self._masked_mean(f_star_of_t_fake-t_real, mask)
        loss = term1 
        return loss

    def generator_loss(self, critic_fake_logits: torch.Tensor) -> torch.Tensor:

        t_fake =  self.conjugate_fn(self.activation_fn(critic_fake_logits),self.epsilon)
        
        loss = -torch.mean(t_fake) 
        return loss
    
    def __repr__(self) -> str:
        return f"FDivergenceLoss(divergence_name='{self.divergence_name}', epsilon={self.epsilon})"
   
