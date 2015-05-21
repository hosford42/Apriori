"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATA_SET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATA_SET.csv -s 0.15 -c 0.6
"""

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def return_items_with_min_support(item_set, transaction_list, min_support, freq_set):
        """calculates the support for items in the item_set and returns a subset
       of the item_set each of whose elements satisfies the minimum support"""
        _itemSet = set()
        local_set = defaultdict(int)

        for item in item_set:
                for transaction in transaction_list:
                        if item.issubset(transaction):
                                freq_set[item] += 1
                                local_set[item] += 1

        for item, count in local_set.items():
                support = float(count)/len(transaction_list)

                if support >= min_support:
                        _itemSet.add(item)

        return _itemSet


def join_set(item_set, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def get_item_set_transaction_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))              # Generate 1-itemsets
    return item_set, transaction_list


def run_apriori(data_iter, min_support, min_confidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pre_tuple, post_tuple), confidence)
    """
    item_set, transaction_list = get_item_set_transaction_list(data_iter)

    freq_set = defaultdict(int)
    large_set = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy min_support

    # Dictionary which stores Association Rules

    one_candidate_set = return_items_with_min_support(
        item_set,
        transaction_list,
        min_support,
        freq_set
    )

    current_length_set = one_candidate_set
    k = 2
    while current_length_set:
        large_set[k-1] = current_length_set
        current_length_set = join_set(current_length_set, k)
        current_candidate_set = return_items_with_min_support(
            current_length_set,
            transaction_list,
            min_support,
            freq_set
        )
        current_length_set = current_candidate_set
        k += 1

    # noinspection PyShadowingNames
    def get_support(item):
            """local function which Returns the support of an item"""
            return float(freq_set[item])/len(transaction_list)

    result_items = []
    for key, value in large_set.items():
        result_items.extend([
            (tuple(item), get_support(item))
            for item in value
        ])

    result_rules = []
    for key, value in large_set.items():
        if key < 2:
            continue
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = get_support(item)/get_support(element)
                    if confidence >= min_confidence:
                        result_rules.append((
                            (tuple(element), tuple(remain)),
                            confidence)
                        )
    return result_items, result_rules


# noinspection PyShadowingNames
def print_results(items, rules):
    """prints the generated itemsets and the confidence rules"""
    for item, support in items:
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ RULES:")
    for rule, confidence in rules:
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def data_from_file(file_name):
        """Function which reads from the file and yields a generator"""
        file_iter = open(file_name, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record


if __name__ == "__main__":

    option_parser = OptionParser()
    option_parser.add_option('-f', '--inputFile',
                             dest='input',
                             help='filename containing csv',
                             default=None)
    option_parser.add_option('-s', '--minSupport',
                             dest='minS',
                             help='minimum support value',
                             default=0.15,
                             type='float')
    option_parser.add_option('-c', '--minConfidence',
                             dest='minC',
                             help='minimum confidence value',
                             default=0.6,
                             type='float')

    (options, args) = option_parser.parse_args()

    inFile = None
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = data_from_file(options.input)
    else:
            print('No data set filename specified, system with exit\n')
            sys.exit('System will exit')

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = run_apriori(inFile, minSupport, minConfidence)

    print_results(items, rules)
