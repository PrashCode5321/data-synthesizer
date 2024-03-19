from scipy.stats import truncnorm
import data_gen_logger
import numpy as np
import logging

logger = logging.getLogger('data_gen')

def get_samples(lower_bound:float, upper_bound:float, mean:float, \
                sd:float, size:int) -> np.array:
    """Performs random sampling on truncated standard distribution. \n
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
        `np.array`: array of generated data
    """    
    try:
        # compute z score of the truncation limits
        z_score_low = (lower_bound - mean) /sd
        z_score_up = (upper_bound - mean) /sd
        # perform random sampling on the truncated normal distribution
        data = truncnorm.rvs(a=z_score_low, b=z_score_up, loc=mean, 
                             scale=sd, size=size)
        return data
    except Exception as e:
        logger.exception(e)
        raise e


def configure_distribution(mean: float, std: float, gradient: float, \
                           thresh: float, size: int, i: int, stage: int)\
                           -> tuple[np.ndarray, float, float]:
    """Configures lower and upper limits for the normal distribution meant
     for generating samples.

    Args:
        `mean` (float): mean of the `parameter`
        `std` (float): standard deviation of the `parameter`
        `gradient` (float): slope for the `parameter` across stages
        `thresh` (float): `parameter` value from the input
        `size` (int): number of samples to generate
        `i` (int): current stage of data synthesis in iterator
        `stage` (int): input stage

    Raises:
        `e`: generic errors/exception

    Returns:
        `tuple[np.ndarray, float, float]`: the data and distribution limits
    """    
    try:
        # if parameter progresses with disease progression
        if gradient > 0:
            lb = mean - 1.5*std if ((mean - 1.5*std) < thresh) else \
                thresh - 1.5*std
            lb = mean*0.2 if lb <= 0 else  lb
            ub = thresh if i == stage else mean + 1.5*std if \
                ((mean + 1.5*std) < thresh) else thresh
            lb = mean*0.1 if lb >= ub else lb
        # if parameter digresses with disease progression
        else:
            lb = thresh if i == stage else mean - 1.5*std if \
                ((mean - 1.5*std) > thresh) else thresh
            lb = mean*0.2 if lb <= 0 else  lb
            ub = mean + 1.5*std if ((mean + 1.5*std) > thresh) \
                else thresh + 2*std
        new_values = get_samples(lower_bound=lb, upper_bound=ub, \
                                 mean=mean, sd=std, size=size)
        return new_values, lb, ub
    except Exception as e:
        logger.exception(e)
        raise e
    
def data_synthesizer(reference: dict, parameter: str, 
                     age_groups: list, adj_size: int, 
                     stage_code: int, thresh: float) -> np.array:
    """Retrieves statistical parameters for the age groups 
    and stages iteratively and uses them for data generation.\n
    This function returns generated data.

    Args:
        `reference` (dict): dictionary containing the statistical parameters
        `parameter` (str): name of the medical test parameter
        `age_groups` (list): computed `ages` mapped to age groups
        `adj_size` (int): number of samples to generate per stage
        `stage_code` (int): encoding of the input stage
        `thresh` (float): upper limit for data generation

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `np.array`: generated data
    """    
    try:
        # fetching all mean values across stages for each age group
        y = {key: [ref.get(f"STAGE-{i+1}", None).get(parameter, None)\
                   .get('mean', None) for i in range(stage_code)] \
                    for key, ref in reference.items()}
        
        # computing gradients of the means for each age group
        grads = {k: np.polyfit(x=np.arange(1, stage_code+1), y=v, deg=1)[0]\
                  for k, v in y.items()}
        
        data = np.array([thresh])
        for stage in range(stage_code, 0, -1):
            batch = age_groups[(stage-1)*adj_size: stage*adj_size]

            # if the mapped age groups are same in the batch,
            # use statistical parameters of that age group only
            if batch.count(batch[0]) == len(batch):
                size = len(batch)
                unique = batch[0]
                params = reference.get(f"AGE GROUP-{unique}")\
                    .get(f"STAGE-{stage}").get(parameter)
                mean, std = params.get("mean"), params.get("std")
                gradient = grads.get(f"AGE GROUP-{unique}")
                values, lb, ub = configure_distribution(mean=mean, std=std, 
                                                        thresh=thresh, 
                                                        size=size, i=stage, 
                                                        stage=stage_code, 
                                                        gradient=gradient)
                values.sort()
                log = (f"AG-{unique} S-{stage}::Generated {size}", 
                       f" data for {parameter}:- m: {mean} | s: {std} | lb: ",
                       f"{lb} | ub: {ub} | thresh: {thresh} | ",
                       f"grad: {gradient}")
                logger.debug(''.join(log))

                # if the parameter progresses with disease progression
                if gradient > 0:
                    data = np.concatenate([values, data])
                    thresh = lb
                # if the parameter digresses with disease progression
                else:
                    values = np.flip(values)
                    data = np.concatenate([values, data])
                    thresh = ub
            
            # if patient's age group changes in a batch,
            # use statistical parameters pertaining age groups iteratively
            else:
                # unique age groups in the batch
                uniques = list({item for item in batch})
                uniques.reverse()
                
                # proportion of age group in this batch 
                sizes = [batch.count(group) for group in \
                         {item for item in batch}]
                sizes.reverse()

                # iterate through each age groups's statistical parameters
                values = []
                for size, unique in zip(sizes, uniques):
                    params = reference.get(f"AGE GROUP-{unique}")\
                        .get(f"STAGE-{stage}").get(parameter)
                    mean, std = params.get("mean"), params.get("std")
                    gradient = grads.get(f"AGE GROUP-{unique}")
                    values1, lb, ub = configure_distribution(
                                                mean=mean, 
                                                std=std, 
                                                thresh=thresh, 
                                                size=size, 
                                                i=stage, 
                                                stage=stage_code, 
                                                gradient=gradient
                                            )
                    log = (f"AG-{unique} S-{stage}::Generated {size}", 
                        f" data for {parameter}:- m: {mean} | s: {std} ",
                        f"| lb: {lb} | ub: {ub} | thresh: {thresh} | ",
                        f"grad: {gradient}")
                    logger.debug(''.join(log))
                    values.append(values1)

                # join data generated using different age groups' statistics
                values = np.concatenate(values)
                values.sort()

                # if the parameter progresses with disease progression
                if gradient > 0:
                    data = np.concatenate([values, data])
                    thresh = lb
                
                # if the parameter digresses with disease progression
                else:
                    values = np.flip(values)
                    data = np.concatenate([values, data])
                    thresh = ub
        return data
    except Exception as e:
        logger.exception(e)
        raise e
