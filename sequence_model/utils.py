from torch.nn import functional as F
import numpy as np
import torch

def radian_l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    https://arxiv.org/abs/2209.15611
    Computes the loss between input and target
    >>> radian_l1_loss(torch.tensor(0.1), 2 * torch.pi)
    tensor(0.1000)
    >>> radian_l1_loss(torch.tensor(0.1), torch.tensor(2 * np.pi - 0.1))
    tensor(0.2000)
    """
    # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
    target = target % (2 * torch.pi)
    input = input % (2 * torch.pi)
    d = target - input
    d = (d + torch.pi) % (2 * torch.pi) - torch.pi
    retval = torch.abs(d)
    return torch.mean(retval)

def radian_smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    circle_penalty: float = 0.0,
) -> torch.Tensor:
    """
    https://arxiv.org/abs/2209.15611
    Smooth radian L1 loss
    if the abs(delta) < beta --> 0.5 * delta^2 / beta
    else --> abs(delta) - 0.5 * beta
    See:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#smooth_l1_loss
    >>> radian_smooth_l1_loss(torch.tensor(-17.0466), torch.tensor(-1.3888), beta=0.1)
    tensor(3.0414)
    """
    assert (
        target.shape == input.shape
    ), f"Mismatched shapes: {input.shape} != {target.shape}"
    assert beta > 0
    d = target - input
    d = modulo_with_wrapped_range(d, -torch.pi, torch.pi)
    abs_d = torch.abs(d)
    retval = torch.where(abs_d < beta, 0.5 * (d**2) / beta, abs_d - 0.5 * beta)
    assert torch.all(retval >= 0), f"Got negative loss terms: {torch.min(retval)}"
    retval = torch.mean(retval)
    # Regularize on "turns" around the circle
    if circle_penalty > 0:
        retval += circle_penalty * torch.mean(
            torch.div(torch.abs(input), torch.pi, rounding_mode="trunc")
        )
    return retval

def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    https://arxiv.org/abs/2209.15611
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max
    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min
    return retval

def exists(val):
    return val is not None

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

def elbo_loss(logits1, logits2, eps=1e-6):
    """
    Compute the Evidence Lower Bound (ELBO) loss.
    
    Args:
        logits1: Logits from the model (shape: (batch_size, length, num_class)).
        logits2: Logits from the variational distribution (shape: (batch_size, length, num_class)).
        eps: Small constant to avoid numerical instability.
    
    Returns:
        ELBO loss value.
    """
    # length, num_class = logits1.shape
    # Convert logits to probabilities
    probs1 = F.softmax(logits1, dim=-1)  # p(x|z)
    probs2 = F.softmax(logits2, dim=-1)  # q(z|x)

    # Calculate KL Divergence
    log_probs1 = F.log_softmax(logits1 + eps, dim=-1)  # Log probabilities of model logits
    log_probs2 = F.log_softmax(logits2 + eps, dim=-1)  # Log probabilities of variational logits
    
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')  # KL(q || p)
    
    # Calculate the negative log-likelihood
    nll = -torch.mean(torch.sum(probs1 * log_probs1, dim=-1))  # Negative log likelihood
    
    # Compute ELBO
    elbo = nll + kl_div  # ELBO loss = NLL + KL divergence
    
    return elbo

# Following codes are modified from: https://arxiv.org/abs/2306.16819
def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)

def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)

def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps
        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif noise_schedule == 'custom':
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)
        sigmas2 = 1 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2     # (timesteps + 1, )
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule_discrete(timesteps)
        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]

class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x

class BlosumTransition:
    def __init__(self, blosum_path='./blosum_substitute.pt',x_classes=20,timestep = 500):
        try:
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        except FileNotFoundError:
            blosum_path = '../'+blosum_path
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        self.X_classes = x_classes
        self.timestep = timestep
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        Qt_temperature = self.Qt_temperature.unsqueeze(dim=0)
        Qt_temperature = Qt_temperature.unsqueeze(dim=0)
        if temperature_list.shape[0] != self.timestep:
            output_tensor = F.interpolate(temperature_list, size=timestep+1, mode='linear', align_corners=True)
            self.temperature_list = output_tensor.squeeze()
            output_tensor = F.interpolate(Qt_temperature, size=timestep+1, mode='linear', align_corners=True)
            self.Qt_temperature = output_tensor.squeeze()
        else:    
            self.temperature_list = self.temperature_list
            self.Qt_temperature = self.Qt_temperature
    
    def get_Qt_bar(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        q_x[q_x < 1e-6] = 1e-6
        return q_x

    def get_Qt(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.Qt_temperature = self.Qt_temperature.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.Qt_temperature[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        return q_x

