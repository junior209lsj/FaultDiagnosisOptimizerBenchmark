from scipy import stats

from typing import List, Dict

def log_qsample(n_params: int,
            param_names: List[str],
            lb: List[float],
            ub: List[float],
            reversed: List[bool],
            n_exps: int,
            seed: int = None) -> Dict:
    """
    Sample hyperparameters in search space from a quasi-random distribution on a log scale.

    The $i^{th}$ hyperparameter $q_{i}$ is sampled from following equation:

    $q_{i} ~ 10^{U[lb_{i}, ub_{i}]}$, if $reversed_{i}$ is False.
    $1 - q_{i} ~ 10^{U[lb_{i}, ub_{i}]}$, if $reversed_{i}$ if True.

    Author: Seongjae Lee

    Parameters
    ----------
    n_params: int
        The number of hyperparameters.
    param_names: List[str]
        The list containing the name of hyperparameters.
    lb: List[float]
        Lower bound of each hyperparameter.
    ub: List[float]
        Upper bound fo each hyperparameter.
    reversed: List[bool]
        Flag determining whether to sample hyperparameters from 1-x.
    n_exps: int
        The number of samples.
    seed: Optional[int]
        Random seed.
    
    Returns
    ----------
    Dict
        Sampled hyperparameters from a log scale quasi-random distribution.
        The key and value of dictionary is the name of the hyperparameter
        and the sampling results, respectively.
    """
    sampler = stats.qmc.Halton(d=n_params, scramble=False, seed=seed)
    samples = sampler.random(n_exps)
    q_samples = stats.qmc.scale(samples, lb, ub)

    param_map = {}

    for i in range(n_params):
        p = 10 ** q_samples[:, i]
        if reversed[i]:
            p = 1 - p
        param_map[param_names[i]] = p
    
    return param_map