def fold(f: bool or list, l: bool or list, a: bool or list) -> bool:
    """ [summary] logical check during parameter folding """
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x: bool or list, y: bool or list) -> bool:
    """ [summary] logical and check """
    return convert(x) and convert(y)


def f_or(x: bool or list, y: bool or list) -> bool:
    """ [summary] logical or check """
    return convert(x) or convert(y)


def convert(x: bool or str) -> bool:
    """[summary] converts parameters too bool """
    if type(x) != bool:
        return len(x) > 0
    else:
        return x


def parameters_allocation_check(module: dict) -> bool:
    """
    [summary] checks if model parameters are allocated correctly
    """
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def handle_inputs(inputs: list, use_cuda: bool) -> list:
    """[summary] transforms a set of inputs to cuda activated inputs to enable GPU"""
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef_mono(iteration: int) -> float:
    """
    [summary] determines the kld_coef_weight for sigmoidal annealing of the kl loss as described in (Bowman  et  al.,  2016)
    """
    import math

    kld_coef_weight = (math.tanh((iteration - 3500) / 1000) + 1) / 2

    return kld_coef_weight


# https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
def kld_coef_cyc(iteration: int, coef_modulo: int) -> float:
    """
    [summary] determines the linear cyclical kld_coef_weight for cyclical annealing (Fu et al. 2019)

    Args:
        iteration ([type]): [description] iteration of training process
        coef_modulo ([type]): [description] step for cyclical  annealing

    Returns:
        [type]: [description] weight for the kld loss
    """
    import math

    if coef_modulo == 0:
        kld_coef_weight = iteration / coef_modulo

        return kld_coef_weight

    try:
        test = math.floor(iteration / coef_modulo)
    except:
        test = 0

    if test % 2 == 0:
        kld_coef_weight = (iteration % coef_modulo) / coef_modulo
    else:
        kld_coef_weight = 1

    return kld_coef_weight
