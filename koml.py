import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, collections as clt
import itertools as it
import argparse as argp

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
    rand_seed = range(10)
    k = []
    for r in rand_seed:
        np.random.seed(r)
        for i in range(loop_ct):
            U = gen_rand_vecs(n)
            k += koml_norm(U, gen_eps(n)),
    return k

def save_fig(k, n, odir):
    fig,ax = plt.subplots()
    ax.scatter(range(len(k)), k, alpha=.3)
    plt.gcf().text(1.,0.8, 'n: ' + str(n), color='r', weight='bold' )
    plt.gcf().text(1.,0.7, 'empirical max: ' + str(np.max(k)), color='r', weight='bold' )
    plt.gcf().text(1.0,0.6, '1 + 1./sqrt(n): ' + str(1 + 1./np.sqrt(n)), color='r' , weight='bold')
    plt.gcf().text(1.0,0.5, 'sqrt(n): ' + str(np.sqrt(n)), color='r' , weight='bold')
    plt.gcf().text(1.0,0.4, 'empirical mean: ' + str(np.mean(k)), color='g' , weight='bold')
    plt.gcf().text(1.0,0.3, 'empirical var: ' + str(np.var(k)), color='g' , weight='bold')
    plt.savefig(odir + '/koml_' + str(n) + '.png', bbox_inches='tight')
    plt.close(fig)

import timeit
if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-outdir', '--out_dir', type=str, help='Output directory', default='/Users/vmullachery/mine/NYU/spring2018/MathDS')
    args = parser.parse_args()

    #Turn off interactive plotting
    plt.ioff()

    start_time = timeit.default_timer()
    loop_ct = 1000
    srange = range(1, 4)
    for i in srange:
        k = compute_koml(i, loop_ct)
        save_fig(k, i, args.out_dir)

    elapsed_time = timeit.default_timer() - start_time
    print(elapsed_time, loop_ct, srange)
