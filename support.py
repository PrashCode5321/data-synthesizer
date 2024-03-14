from scipy.stats import truncnorm
import data_gen_logger
import numpy as np
import logging

logger = logging.getLogger('data_gen')

def get_samples(lower_bound:float, upper_bound:float, 
                mean:float, sd:float, size:int)->np.ndarray:
    """Performs random sampling on truncated standard distribution. 
    Truncation is defined by `lower_bound` and `upper_bound`.

    Args:
        `lower_bound` (float): lower limit on the distribution
        `upper_bound` (float): upper limit on the distribution
        `mean` (float): mean of the standard distribution
        `sd` (float): standard deviation of the standard distribution
        `size` (int): number of points to sample

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `np.ndarray`: array of generated data
    """    
    try:
        z_score_low = (lower_bound - mean) /sd
        z_score_up = (upper_bound - mean) /sd
        data = truncnorm.rvs(a=z_score_low, b=z_score_up, 
                                loc=mean, scale=sd, size=size)
        return data
    except Exception as e:
        logger.exception(e)
        raise e


def generate_data(reference: dict, parameter: str, 
                  stage: int, thresh: float, size: int=25) -> np.ndarray:
    """Generates data for the given parameter `parameter` 
    using the confidence intervals in `reference` dictionary.

    Args:
        `reference` (dict): mean and standard deviation of each parameter
        `parameter` (str): the parameter for which data is being generated
        `stage` (int): current stage as per the input data in encoded form
        `thresh` (float): upper limit for data generation
        `size` (int, optional): number of samples to generate

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `np.ndarray`: generated data with `size` number of samples
    """  
    try:  
        for i in range(stage):
            params = reference.get(f"STAGE-{i+1}", None).get(parameter, None)
            mean = params.get('mean', None)
            std = params.get('std', None)
            if i == 0:
                lb = mean - 2.5*std if (mean - 2.5*std) > mean*0.5\
                    else mean*0.5
                ub = thresh if i == stage-1 else mean + 2*std \
                    if ((mean + 2*std) < thresh) else thresh
                log = [f"{i+1}:::{parameter} - mean: {mean}, std: {std},",
                    f" upper limit: {ub}, lower limit: {lb},", 
                    f" thresh: {thresh}"]
                logger.debug(''.join(log))
                values = get_samples(lower_bound=lb, 
                                    upper_bound=ub, 
                                    mean=mean, sd=std, 
                                    size=size)
            else:
                lb = ub - 2*std if ((mean - 2*std) < (ub - 2*std))\
                     else mean - 2*std
                ub = thresh if i == stage-1 else mean + 2*std \
                    if ((mean + 2*std) < thresh) else thresh
                log = [f"{i+1}:::{parameter} - mean: {mean}, std: {std},",
                    f" upper limit: {ub}, lower limit: {lb},", 
                    f" thresh: {thresh}"]
                logger.debug(''.join(log))
                new_values = get_samples(lower_bound=lb, 
                                        upper_bound=ub, 
                                        mean=mean, sd=std, 
                                        size=size)
                values = np.concatenate([values, new_values])
        values.sort()
        return values
    except Exception as e:
        raise e


