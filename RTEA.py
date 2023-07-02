import numpy as np

def RTEA(evaluations, cost_function, domain_function, l, num_obj, std_mut, p_cross, func_arg):
    if evaluations < 100:
        raise ValueError("As the algorithm samples 100 points initially, evaluations must be at least 100")
    if l < 1:
        raise ValueError("Cannot have zero or fewer design variables")
    if num_obj < 1:
        raise ValueError("Cannot have zero or fewer objectives")
    if std_mut < 0:
        raise ValueError("Mutation width should not be negative")
    if p_cross < 0:
        raise ValueError("Crossover probability should not be negative")

    search_proportion = 0.95
    unique_locations = 100 + np.ceil(((evaluations - 100) / 2) * search_proportion).astype(int)

    X = np.zeros((unique_locations, l))
    Y_mo = np.zeros((unique_locations, num_obj))
    Y_n = np.zeros(unique_locations)
    Y_dom = np.zeros(unique_locations)

    X[0, :] = generate_legal_initial_solution(domain_function, l, func_arg)
    Y_mo[0, :] = evaluate_f(cost_function, X[0, :], num_obj, 1, func_arg)
    Y_n[0] = 1
    I_dom = [0]
    index = 1

    for i in range(1, 100):
        x = generate_legal_initial_solution(domain_function, l, func_arg)
        X, Y_mo, Y_n, Y_dom, index, I_dom = evaluate_and_set_state(
            cost_function, x, num_obj, 1, func_arg, X, Y_mo, Y_n, Y_dom, index, -1, I_dom, i+1
        )

    evals = 100
    std_mut = std_mut * func_arg['range']

    while evals < evaluations:
        if evals <= evaluations * search_proportion:
            x = evolve_new_solution(X, I_dom, domain_function, l, std_mut, p_cross, func_arg)
            X, Y_mo, Y_n, Y_dom, index, I_dom = evaluate_and_set_state(
                cost_function, x, num_obj, 1, func_arg, X, Y_mo, Y_n, Y_dom, index, -1, I_dom, evals+1
            )
            evals += 1

        if evals < evaluations:
            _, II = min(Y_n[I_dom])
            copy_index = I_dom[II[0]]
            x = X[copy_index, :]
            X, Y_mo, Y_n, Y_dom, index, I_dom = evaluate_and_set_state(
                cost_function, x, num_obj, 1, func_arg, X, Y_mo, Y_n, Y_dom, index, copy_index, I_dom, evals+1
            )
            evals += 1

    X = X[:index, :]
    Y_mo = Y_mo[:index, :]
    Y_dom = Y_dom[:index]
    Y_n = Y_n[:index]

    return I_dom, X, Y_mo, Y_n, Y_dom

def evaluate_and_set_state(cost_function, x, num_obj, initial_reevals, func_arg, X, Y_mo, Y_n, Y_dom, index,
                           prev_index, I_dom, evals):
    x_objectives = evaluate_f(cost_function, x, num_obj, initial_reevals, func_arg)

    if initial_reevals == 1:
        Y_mo[index, :] = x_objectives
        Y_dom[index] = 0
        I_dom = check_dominance(index, Y_mo, Y_n, Y_dom, I_dom)
        if Y_dom[index] == 0:
            X[index, :] = x
            Y_n[index] = 1
            index += 1
    else:
        Y_mo[prev_index, :] = x_objectives
        Y_dom[prev_index] = 0
        I_dom = check_dominance(prev_index, Y_mo, Y_n, Y_dom, I_dom)
        if Y_dom[prev_index] == 0:
            X[prev_index, :] = x
            Y_n[prev_index] = 1

    return X, Y_mo, Y_n, Y_dom, index, I_dom

def evaluate_f(cost_function, x, num_obj, num_repeats, func_arg):
    x_objectives = np.zeros(num_obj)
    for i in range(num_repeats):
        x_objectives += cost_function(x, func_arg)
    x_objectives /= num_repeats
    return x_objectives

def check_dominance(index, Y_mo, Y_n, Y_dom, I_dom):
    for i in I_dom:
        if Y_n[i] == 0:
            dom = 0
            for j in range(Y_mo.shape[1]):
                if Y_mo[index, j] > Y_mo[i, j]:
                    dom = -1
                    break
                if Y_mo[index, j] < Y_mo[i, j]:
                    dom = 1
            if dom == -1:
                Y_n[i] = 1
            if dom == 1:
                Y_n[index] = 1
                Y_dom[index] += 1
                if Y_dom[i] == 0:
                    I_dom.remove(i)
    if Y_dom[index] == 0:
        I_dom.append(index)
    return I_dom

def generate_legal_initial_solution(domain_function, l, func_arg):
    x = np.zeros(l)
    for i in range(l):
        x[i] = domain_function(i, func_arg)
    return x

def evolve_new_solution(X, I_dom, domain_function, l, std_mut, p_cross, func_arg):
    pop_size = len(I_dom)
    K = np.random.choice(I_dom, pop_size)
    W = np.random.normal(0, std_mut, (pop_size, l))
    R = np.random.choice([0, 1], (pop_size, l), p=[1 - p_cross, p_cross])

    x = np.zeros(l)
    for i in range(l):
        w = W[:, i]
        r = R[:, i]
        k1 = K[np.random.randint(pop_size)]
        k2 = K[np.random.randint(pop_size)]
        x[i] = X[k1, i] + w[k1] * (X[k2, i] - X[k1, i]) + w[k2] * (X[k2, i] - X[k1, i]) * r[k1]
        x[i] = domain_function(i, func_arg, x[i])

    return x
