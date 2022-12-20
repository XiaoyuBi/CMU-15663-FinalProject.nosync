import numpy as np

def conjgrad(x, b, Ax_func, Ax_param, maxIter = 20, tol = 1e-5):
    """
    Conjugate Gradient Method 

    For solving "x" from "Ax = b" efficiently
    Or solving "x" from "Ax_func(x, Ax_param) = b"

    Input:
        x: initial value of x
        b: target value
        Ax_func, Ax_param:
        maxIter, tol: max iteration and tolerance for stopping condition
    Output:
        x: solution x for "Ax = b"
    """

    # residual
    r = b - Ax_func(x, Ax_param)
    # conjugate vector
    p = r

    rsold = np.sum(r * r)

    for _ in range(maxIter):
        Ap = Ax_func(p, Ax_param)
        alpha = rsold / np.sum(p * Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        
        p = r + rsnew / rsold * p
        rsold = rsnew
    
    return x