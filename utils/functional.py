def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))

def f_and(x, y):
    return convert(x) and convert(y)

def f_or(x, y):
    return convert(x) or convert(y)

def convert(x):
    if type(x) != bool:
        return len(x)>0
    else:
        return x

def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)

def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef_mono(iteration):
    import math
    return (math.tanh((iteration - 3500)/1000) + 1)/2

# https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
def kld_coef_cyc(iteration, coef_modulo):
    import math

    if coef_modulo == 0:
        return iteration / coef_modulo

    try:
        test = math.floor(iteration / coef_modulo)
    except:
        test = 0

    if test % 2 == 0:
        return (iteration % coef_modulo)/coef_modulo
    else:
        return 1
