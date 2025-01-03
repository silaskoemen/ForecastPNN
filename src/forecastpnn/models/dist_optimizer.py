import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import skewnorm
from torch.distributions import Normal
from tqdm import tqdm


def negative_log_likelihood(data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    # Compute the negative log likelihood of the skewnormal distribution
    alpha, loc, scale = params[0], params[1], params[2]
    # Add penalty for invalid parameter values
    if scale <= 0:
        return 1e200
    # Divide pdf by cdf at bound
    normal_dist = Normal(loc, scale)
    normalized_data = (data - loc) / scale
    probs = 2/scale*normal_dist.log_prob(normalized_data).exp() * normal_dist.cdf(alpha*normalized_data)
    exp = probs * data
    cond_exp = exp/(skewnorm.cdf(data.max(), alpha.item(), loc.item(), scale.item()) + 1e-10)
    #cond_exp = exp/torch.tensor((skewnorm.cdf(data.max(), alpha.detach().numpy(), loc.detach().numpy(), scale.detach().numpy()) + 1e-10))
    return -torch.sum(torch.log(cond_exp + 1e-10))


def calculate_m0(params):
    delta = params[0] / (torch.sqrt(1 + params[0]**2))
    pi_tensor = torch.tensor(torch.pi)
    m0 = (
        torch.sqrt(2/pi_tensor) * delta
        - (1 - pi_tensor / 4) * (torch.sqrt(2 / pi_tensor) * delta)**3
        / (1 - 2 / pi_tensor * delta**2)
        - torch.sign(params[0]) / 2 * torch.exp(-2 * pi_tensor / abs(params[0]))
    )
    return m0


def total_loss(data, params, peak_loc, lambda_loc, lambda_alpha: float = None, alpha_bounds = (-10, 10)):
    nll = negative_log_likelihood(data, params)
    m0 = calculate_m0(params)
    if lambda_alpha is not None:
        if params[0].item() < alpha_bounds[0] or params[0].item() > alpha_bounds[1]:
            #print(f"Alpha out of bounds: {params[0]}")
            return nll + lambda_loc * torch.abs(params[1] + params[2] * m0 - peak_loc) + lambda_alpha * torch.abs(params[0]-alpha_bounds[0])
    return nll + lambda_loc * torch.abs(params[1] + params[2] * m0 - peak_loc)


def optimize_params(data, peak_loc, lr=0.1, n_iter=1000, lambda_loc = 1e4, lambda_alpha = None, alpha_bounds = (-10, 10)):
    # Cast the data to a tensor
    data = torch.tensor(data, dtype=torch.float32)
    mu0, sig0 = data.mean().item(), data.std().item()
    
    # Define the parameters of the skewnormal distribution
    params = nn.Parameter(torch.tensor([0.1, mu0, sig0]), requires_grad=True)
    optimizer = Adam([params], lr=lr)

    for i in tqdm(range(n_iter)):
    #for i in range(n_iter):
        #print(f"Start of iteration {i} - params: {params} -  data: {data}")
        optimizer.zero_grad()
        loss = total_loss(data, params, peak_loc, lambda_loc, lambda_alpha=lambda_alpha, alpha_bounds=alpha_bounds)
        loss.backward()
        optimizer.step()
        #print(f"End of iteration {i} - params: {params} - Loss: {loss.item()} - data: {data}")

    return params.detach().numpy()