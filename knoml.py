import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, collections as clt
import itertools as it
import argparse as argp
import pickle

np.random.seed(81977)
odir = ''
threshold=1.59

def gen_rand_vecs(n):
    dims = n
    number = n
    vecs = np.random.normal(size=(number, dims))
    mags = np.linalg.norm(vecs, axis=-1)

    return vecs/mags[...,np.newaxis]

def plot_lines(ends):
    # ends = gen_rand_vecs(2)
    vectors = np.insert(ends[:, np.newaxis], 0, 0, axis=1)
    fig, ax = plt.subplots()
    ax.add_collection(clt.LineCollection(vectors))
    ax.axis((-1,1,-1,1))
    plt.show()

def gen_eps(n):
    eps_vals = [-1, 1]
    combs = [ x for x in it.combinations_with_replacement(eps_vals, n) ]
    return get_half_set(set([ x for t in combs for x in it.permutations(t) ]))

def koml_norm(U, eps_vecs):
    w = []
    for eps in eps_vecs:
        v = np.transpose(U).dot(eps)
        w += np.max(abs(v)),
    return np.min(w)

def compute_koml(n, loop_ct):
    '''
    Side effects: saved figure of least maxnorm,
    saved pickle file of n, n-dimension vectors whose least maxnorm with epsilons were gt 1.5
    :param n: dimension
    :param loop_ct: number of random matrices to try
    :return: k (vector of least maxnorm), H (least maxnorm, contributing n-dim vectors when gt 1.5)
    '''
    k = []
    H = []
    eps_vecs = gen_eps(n)
    for i in range(loop_ct):
        U = gen_rand_vecs(n)
        nm = koml_norm(U, eps_vecs)
        if nm > threshold:
            k.append(nm)
            H.append((nm, U))
            save_fig(k, n, odir)
            save_hivecs(H, n, odir)

    return k, H

def save_fig(k, n, odir):
    fig,ax = plt.subplots()
    ax.scatter(range(len(k)), k, alpha=.3)
    plt.gcf().text(1.,0.8, 'n: ' + str(n), color='r', weight='bold' )
    plt.gcf().text(1.,0.7, 'empirical max: ' + str(np.max(k)), color='r', weight='bold' )
    plt.gcf().text(1.0,0.6, '1 + 1./sqrt(n): ' + str(1 + 1./np.sqrt(n)), color='r' , weight='bold')
    plt.gcf().text(1.0,0.5, 'sqrt(n): ' + str(np.sqrt(n)), color='r' , weight='bold')
    plt.gcf().text(1.0,0.4, 'empirical mean: ' + str(np.mean(k)), color='g' , weight='bold')
    plt.gcf().text(1.0,0.3, 'empirical var: ' + str(np.var(k)), color='g' , weight='bold')
    plt.savefig(odir + './koml_' + str(n) + '.png', bbox_inches='tight')
    plt.close(fig)

def save_hivecs(H, n, odir):
    fname = odir + './arr_' + str(n) + '.txt'
    pickle.dump(H, open(fname, 'w'))


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

import timeit
if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-outdir', '--out_dir', type=str, help='Output directory', default='/Users/vmullachery/mine/Udacity/LinearAlgebra/MathDS/')
    parser.add_argument('-n', '--dim', type=int, help='Dimension', default=3)
    parser.add_argument('-loops', '--nloops', type=int, help='Number of n-dim matrices to try', default=1000000)
    args = parser.parse_args()

    #Turn off interactive plotting
    plt.ioff()

    odir = args.out_dir
    n = args.dim
    loops = args.nloops
    '''
    n=4 loops       Run times
    ========================
    1000        0.35 secs
    10000       1.26 secs
    100000      10.11 secs
    1000000     105.82 secs
    10000000    4422.91 secs
    '''
    start_time = timeit.default_timer()
    k, H = compute_koml(n, loops)
    save_fig(k, n, odir)
    save_hivecs(H, n, odir)
    elapsed = timeit.default_timer() - start_time
    print elapsed
