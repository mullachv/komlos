import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, collections as clt
import itertools as it
import argparse as argp
import pickle

np.random.seed(7)

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
    return set([ x for t in combs for x in it.permutations(t) ])

def koml_norm(U, eps_vecs):
    w = []
    for eps in eps_vecs:
        v = np.transpose(U).dot(eps)
        w += np.max(abs(v)),
    return np.min(w)

def compute_koml(n, loop_ct):
    k = []
    H = []
    eps_vecs = gen_eps(n)
    for i in range(loop_ct):
        U = gen_rand_vecs(n)
        nm = koml_norm(U, eps_vecs)
        k.append(nm)
        if nm > 1.5:
            H.append((nm, U))
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

#### 2nd half ####
def to_cartesian(R, THETA, PHI):
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    X = np.ndarray.flatten(X)
    Y = np.ndarray.flatten(Y)
    Z = np.ndarray.flatten(Z)
    return zip(X, Y, Z)

def vec_comps():
    n1, n2 = 30, 10
    theta, phi = np.linspace(0, 2 * np.pi, n1), np.linspace(0, np.pi, n2)
    THETA, PHI = np.meshgrid(theta, phi)
    return to_cartesian(1, THETA, PHI)

import itertools
def vecs_3Dspan():
    V = vec_comps()
    print('len V: ', len(V))
    items = itertools.combinations(range(len(V)), 3)
    U = []
    for t in items:
        U += (V[t[0]], V[t[1]], V[t[2]]),
    return U

def compute_koml3():
    n = 3
    k = []
    H = []
    eps_vecs = gen_eps(n)
    for U in vecs_3Dspan():
        nm = koml_norm(U, eps_vecs)
        if nm > 1.5:
            k.append(nm)
            H.append((nm, U))
    print('len k: ', len(k))
    return k, H


import timeit
if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-outdir', '--out_dir', type=str, help='Output directory', default='/Users/vmullachery/mine/Udacity/LinearAlgebra/MathDS/')
    args = parser.parse_args()

    #Turn off interactive plotting
    plt.ioff()

    start_time = timeit.default_timer()
    k, H = compute_koml3()
    save_fig(k, 3, args.out_dir)
    save_hivecs(H, 3, args.out_dir)
    elapsed = timeit.default_timer() - start_time
    print elapsed

