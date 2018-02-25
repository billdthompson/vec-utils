# coding: utf-8

# Author:
# -----------
# Copyright (c) 2018 - present Bill Thompson (billdthompson@berkeley.edu) 

import numpy as np
import pandas as pd
import skipgram

import click
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

VECTORDIMENSIONS = ['d_{0}'.format(d) for d in range(skipgram.D)]

@click.command()
@click.option('--vecfile', '-v', default='wiki.en.vec')
@click.option('--wordlist', '-w', default='wordlist.csv')
def run(vecfile, wordlist):

    # find the semantic model
    logging.info("# Computing Semantic Network")
    logging.info("Preprocessing > Retrieving semantic model from {}".format(vecfile))
    model = skipgram.Skipgram(vecfile)

    # find the word list
    logging.info("Preprocessing > Reading words from {}".format(wordlist))
    wordlist = pd.read_csv(wordlist)[:100]
    wordlist = wordlist[wordlist.word.isin(model.word.index.values)] 

    # find the resource intersection
    logging.info("Preprocessing > Reading vectors")
    vectors = wordlist.word.apply(lambda w: pd.Series(model[w], index = VECTORDIMENSIONS)).values

    # compute Euclidean magnitudes
    logging.info("Compute > Precomputing constants")
    squaresumroots = np.sqrt(np.power(vectors, 2).sum(axis = 1))
    netsize = vectors.shape[0]

    def getstats(row):
        return pd.Series((np.dot(vectors, vectors[row.idx]) / (squaresumroots[row.idx] * squaresumroots)), index = wordlist.word.values)

    logging.info("Compute > Looping over network (# words = {0}; # pairwise = {1}; # unique non-identical pairs = {2})".format(netsize, np.power(netsize, 2), (np.power(netsize, 2) - netsize) / 2.))
    results = pd.DataFrame(dict(word = wordlist.word.values, idx = range(netsize))).apply(getstats, axis = 1)
    
    logging.info("Save > writing network in matrix form to: network-matrix.txt")
    np.savetxt('network-matrix.txt', results.values)

    logging.info("Save > writing col / row index to: network-index.csv")
    wordlist.word.to_csv('network-index.csv')

    logging.info("Save > writing network in long form (unique non-matching node-pairs) to: network-longform.csv")
    i, j = np.triu_indices(netsize, 1)
    longform = pd.DataFrame(dict(w1 = wordlist.word.values[i], w2 = wordlist.word.values[j], cosine_similarity = results.values[i, j].flatten()))
    longform.to_csv('network-longform.csv', index = False, encoding = 'utf-8')
    logging.info("Algorithm complete.")

if __name__ == '__main__':
    run()








