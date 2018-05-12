import numpy as np
import itertools as it
import scipy as sp

np.random.seed(7)

def gen_rand_vecs(n):
    dims = n
    number = n
    vecs = np.random.normal(size=(number, dims))
    mags = np.linalg.norm(vecs, axis=-1)

    return vecs / mags[..., np.newaxis]

def gen_eps(n):
    eps_vals = [-1, 1]
    combs = [x for x in it.combinations_with_replacement(eps_vals, n)]
    return get_half_set(set([x for t in combs for x in it.permutations(t)]))

def get_half_set(s):
    '''
    If s = {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    then return is one of the following:
    {(1, 1), (-1, 1)}
    {(-1, -1), (1, -1)}
    {(1, 1), (1, -1)}
    {(-1, -1), (-1, 1)}

    symmetric reflections are not returned in a single set,
    so (1, 1) and (-1, -1) cannot co-occur in the return
    '''
    p = set([])
    for i in s:
        if i in p:
            continue
        if neg(i) in p:
            continue
        p.add(i)
    return p


def neg(i):
    '''
    Negates the items of a tuple
    if i = (-1, 1), return (1, -1)
    '''
    return tuple([-x for x in i])


import numpy as np
from scipy.optimize import minimize

eps_vecs = gen_eps(3)
print(eps_vecs)


def convert2xyz(phi, theta):
    '''
    return x, y, z from phi and theta. r is 1
    '''
    return np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)


def f0(x):
    g = 1e9
    for eps in eps_vecs:
        g = min(g, g0(x, eps))
    return -g


def g0(x, epsilon):
    '''
    return the infinity norm (max of x, y, z of the sum)
    '''
    #print('x: ', x)
    alpha1, alpha2 = x[0], x[1]
    beta1, beta2 = x[2], x[3]
    gamma1, gamma2 = x[4], x[5]
    one = np.cos(alpha1) * epsilon[0] + np.cos(beta1) * epsilon[1] \
          + np.cos(gamma1) * epsilon[2]
    two = np.sin(alpha1) * np.cos(alpha2) * epsilon[0] \
          + np.sin(beta1) * np.cos(beta2) * epsilon[1] + np.sin(gamma1) * np.cos(gamma2) * epsilon[2]
    three = np.sin(alpha1) * np.sin(alpha2) * epsilon[0] \
            + np.sin(beta1) * np.sin(beta2) * epsilon[1] + np.sin(gamma1) * np.sin(gamma2) * epsilon[2]
    m = max(abs(one), abs(two), abs(three))
    #print ('max: ', m)
    return m

import argparse as argp
import timeit
from scipy import optimize

if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-n', '--inv_step', type=float, help='Inverse Step size', default=2.)
    parser.add_argument('-r', '--ranges', nargs='+', type=float, help='Ranges', default=[0, np.pi, 0, 2*np.pi]*3)
    args = parser.parse_args()

    inv_step = args.inv_step
    ranges_in = args.ranges

    start_time = timeit.default_timer()
    #ranges=(slice(0, np.pi, np.pi/inv_step), slice(0, 2*np.pi, np.pi/inv_step))*3
    ranges = (slice(ranges_in[0], ranges_in[1], abs(ranges_in[1] - ranges_in[0])/inv_step),
              slice(ranges_in[2], ranges_in[3], abs(ranges_in[3] - ranges_in[2])/(2.*inv_step)),
              slice(ranges_in[4], ranges_in[5], abs(ranges_in[5] - ranges_in[4])/inv_step),
              slice(ranges_in[6], ranges_in[7], abs(ranges_in[7] - ranges_in[6])/(2.*inv_step)),
              slice(ranges_in[8], ranges_in[9], abs(ranges_in[9] - ranges_in[8])/inv_step),
              slice(ranges_in[10], ranges_in[11], abs(ranges_in[11] - ranges_in[10])/(2.*inv_step)),
              )
    print(ranges)

    resbrute = optimize.brute(f0, ranges, full_output=True)
    print('Min points: ', resbrute[0], ', Min value: ', resbrute[1])
    elapsed = timeit.default_timer() - start_time
    print('Time: ', elapsed)

#-1.5764358611132199, .9
#-1.5766293274178389, .7
#-1.5714085573238803, .6

