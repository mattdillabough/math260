# Matthew Dillabough - HW10 - 10/6/2020
# Worked with Adi Pall on this assignment

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def rk4(func, interval, y0, h):
    """ RK4 method, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            tvals - a list of the times for the computed solution
            yvals - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)
    t = interval[0]
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]

    while t < interval[1] - 1e-12:
        f0 = func(t, y)
        f1 = func(t + 0.5*h, y + 0.5*h*f0)
        f2 = func(t + 0.5*h, y + 0.5*h*f1)
        f3 = func(t + h, y + h*f2)
        y += (1.0/6)*h*(f0 + 2*f1 + 2*f2 + f3)
        t += h

        for k in range(len(y)):
            yvals[k].append(y[k])

        tvals.append(t)

    return tvals, yvals


def read_sir_data(fname):
    """ Reads the SIR data and outputs it in appropriate form
        Inputs:
            fname - the filename (should be "sir_data.txt")
        Returns:
            t, x - the data (t[k], I[k]), where t=times, I= # infected
            pop - the initial susceptible population (S(0))
    """
    with open(fname, 'r') as fp:
        parts = fp.readline().split()
        pop = float(parts[0])
        npts = int(float(parts[1]))
        t = np.zeros(npts)
        x = np.zeros(npts)

        for k in range(npts):
            parts = fp.readline().split()
            t[k] = float(parts[0])
            x[k] = float(parts[1])

    return t, x, pop


def SIR(x, params):
    """ SIR function as defined in problem statement """
    p = x[0]+x[1]+x[2]
    dS = -1*params[0]*x[0]*x[1]/p
    dI = params[0]*x[0]*x[1]/p - params[1]*x[1]
    dR = params[1]*x[1]

    return np.array((dS, dI, dR))


def err(r):
    """ Least-squares error E(r) for SIR """
    err = 0
    x0 = np.array((100.0, 5.0, 0.0))
    h = 0.1
    _, x = rk4(lambda t, x: SIR(x, r),
               (0, 140), x0, h)

    for k in range(len(data)):
        err += (x[1][::67][k] - data[k])**2

    return err


def minimize(f, r0, tol=1e-6, d=0.01, steps=100):
    """ Minimizes the r tuple
    Inputs:
        f - the function f(x)
        r0 - starting r tuple value
    Returns:
        r - minimized r tuple
        it - number of iterations required to yield r
    """
    r = np.array(r0)
    v = np.zeros(2)
    it = 0
    err_s = 100

    while err_s > tol and it < steps:
        fx = f(r)
        alpha = 1

        v[0] = (f((r[0]+d, r[1]))-f((r[0]-d, r[1]))) / \
            (2*d)  # Calculate first component of gradient
        v[1] = (f((r[0], r[1]+d))-f((r[0], r[1]-d))) / \
            (2*d)  # Calculate second component of gradient
        v /= max(np.abs(v))

        while (r[0] - alpha*v[0]) < 0 or (r[1] - alpha*v[1]) < 0:
            alpha /= 2

        while alpha > 1e-10 and f(r - alpha*v) >= fx:
            alpha /= 2

        r -= alpha*v
        err_s = max(np.abs(alpha*v))
        it += 1

    return r, it


if __name__ == '__main__':
    tvals, data, pop = read_sir_data('sir_data.txt')
    plt.figure(1)
    plt.plot(tvals, data, '.k', markersize=12)
    plt.show()
    # Plots the data values

    alpha = 0.1
    beta = 0.03
    # These values are best fit for I(t) curve
    x0 = np.array((100.0, 5.0, 0.0))
    h = 0.1
    t, x = rk4(lambda t, x: SIR(x, (alpha, beta)),
               (0, 140), x0, h)

    plt.figure(2)
    plt.plot(t, x[0], '-k', t, x[1], '-r', t, x[2], '-g')
    plt.xlabel('t')
    plt.legend(['S(t)', 'I(t)', 'R(t)'])
    plt.show()
    # Plots the trends for S, I, and R

    r0 = (0.2, 0.05)
    r_model, it = minimize(lambda r:  err(r),
                           r0, tol=1e-6)
    h = (tvals[1] - tvals[0])/(2**3)
    t, x = rk4(lambda t, x: SIR(x, (r_model)),
               (0, 140), x0, h)
    plt.figure(3)
    plt.plot(t, x[0], '-k', t, x[1], '-r', t, x[2],
             '-g', tvals, data, '.k', markersize=8)
    plt.legend(['S(t)', 'I(t)', 'R(t)', 'Data'])
    plt.show()
    # Plots the SIR trends with our r_model superimposed
