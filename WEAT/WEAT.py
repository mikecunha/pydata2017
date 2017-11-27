import numpy as np
from scipy.misc import comb

import functools
from itertools import combinations
from random import shuffle

CACHE_SIZE = 2**8  # 256


class WEAT(object):
    """Word Embedding Association Test"""

    def __init__(self, model):
        # pass in a an embedding as a gensim object
        self.model = model
        self.stopping_early = False

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def gensim_cosine(self, a, b):
        return self.model.similarity(a, b)

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cosine_sim_many(self, word_a, other_words):
        """Compare one word to many
        NOTE* similarites will not be returned in the same order that were
        passed into this function because they are sets... make sure you don't
        do anything with the values returned in which order matters."""
        return frozenset([self.gensim_cosine(word_a, word) for
                          word in other_words])

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def mean_diff(self, A, B):
        """helper function for simple calculation of the
        difference between means that can be cached."""
        return sum(A)/len(A) - sum(B)/len(B)

    def partitions_gen(self, target_X, target_Y):
        """Generator of unique partitions in which order of a subset
        does not matter."""

        targets = target_X.union(target_Y)
        group_size = int(len(targets) / 2)
        assert(len(target_X) == len(target_Y) == group_size)

        # make sure we observe the given partition (need this for early
        # stopping in the future)
        observed = tuple(sorted(list(target_X)) + sorted(list(target_Y)))
        seen = {observed}
        yield observed

        # now sample from the other partitions
        t_list = list(targets)

        if self.stopping_early:
            # When stopping early it's important that the combinations we
            # do yield aren't sorted and predictable as they are from
            # itertools combinations... we need randomness to sample the
            # full space of partitions better
            seen_count = 1
            for i in range(self.max_iters*2):
                shuffle(t_list)
                new_X = t_list[:group_size]
                new_Y = t_list[group_size:]
                new_partition = tuple(sorted(new_X)+sorted(new_Y))

                if seen_count >= self.max_iters:
                    break

                # as we see more we'll waste more time here and fail
                # to yield the max_iters requested, that's why we do
                # double the max_iters and keep track of num yielded
                if new_partition in seen:
                    continue

                seen.add(new_partition)
                seen_count += 1
                yield new_partition
        else:
            for c in combinations(t_list, group_size):
                new_X = set(c)
                new_Y = targets.difference(new_X)
                new_partition = tuple(sorted(list(new_X))+sorted(list(new_Y)))

                if new_partition in seen:
                    continue

                seen.add(new_partition)

                yield new_partition

    def permutation_test_stat(self, target_X, target_Y, attr_A, attr_B,
                              skip_effect=True):
        """Calculates statistic for a specific permutation"""
        x_assoc = 0
        y_assoc = 0
        diffs = []  # needed to calc std

        for x, y in zip(target_X, target_Y):

            x_sim_A = self.cosine_sim_many(x, attr_A)
            x_sim_B = self.cosine_sim_many(x, attr_B)
            x_diff = self.mean_diff(x_sim_A, x_sim_B)
            x_assoc += x_diff
            diffs.append(x_diff)

            y_sim_A = self.cosine_sim_many(y, attr_A)
            y_sim_B = self.cosine_sim_many(y, attr_B)
            y_diff = self.mean_diff(y_sim_A, y_sim_B)
            y_assoc += y_diff
            diffs.append(y_diff)

        test_stat = x_assoc - y_assoc

        if skip_effect:
            return test_stat

        # Calculate Effect Size
        std = np.std(diffs)
        mean_assoc_X = x_assoc / len(target_X)
        mean_assoc_Y = y_assoc / len(target_Y)

        effect_size = (mean_assoc_X - mean_assoc_Y) / std

        return test_stat, effect_size

    def check_inputs(self, target_X, target_Y, attr_A, attr_B, max_iters):
        """Perform checks to make sure WEAT inputs adhere to the
        constraints of the problem and that the permutation test
        isn't intractable."""

        targets = target_X.union(target_Y)
        group_size = int(len(targets) / 2)
        err = 'Target word sets must be of equal size and not have repeats.'
        assert(len(target_X) == len(target_Y) == group_size), err

        warn = ''
        n_combs = int(comb(len(targets), group_size))

        if max_iters is None:
            max_iters = n_combs

        if n_combs > max_iters:
            self.stopping_early = True
            self.max_iters = max_iters
            warn += ''.join(['Warning: the P-Value returned may not be ',
                             'trustworthy because all combinations of ',
                             'target words will not be checked (max_iters',
                             '='+str(max_iters)+' is less than the ',
                             '{:,}'.format(n_combs)+' possible ',
                             'combinations)\n\n'])

        A_LOT_OF_ITERS = 50000  # this takes about 30 sec on dev machine

        if max_iters > A_LOT_OF_ITERS:
            warn += ''.join(['Warning: processing the ',
                             '{:,}'.format(max_iters)+' ',
                             'combinations of target words when ',
                             'calculating the P-Value may take a while.',
                             '\n\n'])

        if warn == '':
            warn = None
        return max_iters, warn

    def assert_vocab(self, target_X, target_Y):
        """Assert that the input target words are in the vocabulary
        of the embeding, exclude terms that arent' and make sure
        the two target sets are still balanced."""

        oov_X = set([])
        for term in target_X:
            if not self.model.vocab.get(term):
                oov_X.add(term)

        oov_Y = set([])
        for term in target_Y:
            if not self.model.vocab.get(term):
                oov_Y.add(term)

        if len(oov_X.union(oov_Y)) == 0:
            return target_X, target_Y
        elif len(oov_X) == len(oov_Y):
            pass
        elif len(oov_X) > len(oov_Y):
            # remove additional target_Y terms to balance the set
            # FIXME - randomize?
            delta = len(oov_X) - len(oov_Y)
            for _ in range(delta):
                excl = target_Y.difference(oov_Y).pop()
                target_Y = target_Y.difference(set([excl]))
                print('Target sets are unbalanced, excluding:',
                      excl, 'from target_Y')
        else:
            delta = len(oov_Y) - len(oov_X)
            for _ in range(delta):
                excl = target_X.difference(oov_X).pop()
                target_X = target_X.difference(set([excl]))
                print('Target sets are unbalanced, excluding:',
                      excl, 'from target_X')

        print('Warning: target words not in embedding vocabulary are',
              'being excluded: ', str(oov_X), str(oov_Y))
        if len(oov_X.union(oov_Y)) > min(len(target_X), len(target_Y)):
            print('Warning: more than half your target words are',
                  'out-of-vocabulary and will not be included in the test.')

        return target_X.difference(oov_X), target_Y.difference(oov_Y)

    def perform_test(self, target_X, target_Y, attr_A, attr_B,
                     max_iters=50000):
        """Word Embedding Association Test"""

        # in case embedding has changed since last test
        self.cosine_sim_many.cache_clear()

        attr_A = frozenset(attr_A)
        attr_B = frozenset(attr_B)

        # remove any input words not in the embedding vocab
        target_X, target_Y = self.assert_vocab(target_X, target_Y)

        max_iters, warns = self.check_inputs(target_X, target_Y, attr_A,
                                             attr_B, max_iters)
        if warns:
            print(warns)

        # Calculate observed test-statistic and effect size
        T_obs, effect = self.permutation_test_stat(target_X, target_Y, attr_A,
                                                   attr_B, skip_effect=False)

        # Now calculate test statistics for different groupings of target words
        T_sampled = []
        half = len(target_X)
        assert(len(target_X) == len(target_Y))

        # TODO: there are formulas to stop early and put estimates on the
        # p-value, this will be necessary for large target lists.
        for i, p in enumerate(self.partitions_gen(target_X, target_Y)):
            targ_X = p[:half]
            targ_Y = p[half:]
            T_sampled.append(self.permutation_test_stat(targ_X, targ_Y,
                                                        attr_A, attr_B))

            if i+1 == max_iters:
                # print('{:,} partitions processed.'.format(len(T_sampled)))
                break

        # total observation with statistic >= observed value
        n = sum(t >= T_obs for t in T_sampled)

        # p-value is how often the test statistic was >= observed
        # in a random grouping of the target words
        p_val = n / len(T_sampled)

        return effect, p_val
