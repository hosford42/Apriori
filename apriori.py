"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATA_SET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATA_SET.csv -s 0.15 -c 0.6
"""

import sys

from itertools import chain, combinations
from collections import defaultdict


def subsets(arr):
    """Return non-empty subsets of arr"""
    return chain(*(combinations(arr, i + 1) for i in range(len(arr))))


def get_items_with_min_support(item_set, transaction_list, min_support, freq_set):
    """Calculates the support for items in the itemset and returns a subset
    of the itemset each of whose elements satisfies the minimum support."""
    results = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transaction_list:
            if item <= transaction:
                freq_set[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = count / len(transaction_list)

        if support >= min_support:
            results.add(item)

    return results


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return {i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length}


def get_item_set_transaction_list(data_iterator):
    transaction_list = []
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))              # Generate 1-itemsets
    return item_set, transaction_list


def run_apriori(data_iter, min_support, min_confidence):
    """
    Run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pre_tuple, post_tuple), confidence)
    """
    item_set, transaction_list = get_item_set_transaction_list(data_iter)

    freq_set = defaultdict(int)
    large_set = {}
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy min_support

    # Dictionary which stores Association Rules

    current_length_set = get_items_with_min_support(
        item_set,
        transaction_list,
        min_support,
        freq_set
    )

    k = 2
    while current_length_set:
        print(k)
        large_set[k-1] = current_length_set
        current_length_set = join_set(current_length_set, k)
        current_candidate_set = get_items_with_min_support(
            current_length_set,
            transaction_list,
            min_support,
            freq_set
        )
        current_length_set = current_candidate_set
        k += 1

    # noinspection PyShadowingNames
    def get_support(item):
        """Local function which returns the support of an item"""
        return freq_set[item] / len(transaction_list)

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
            for subset in subsets(item):
                subset = frozenset(subset)
                remain = item.difference(subset)
                if len(remain) > 0:
                    confidence = get_support(item) / get_support(subset)
                    if confidence >= min_confidence:
                        result_rules.append((
                            (tuple(subset), tuple(remain)),
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


def data_from_file(file, ordered=False, ignore=None):
    """Function which reads from the file and yields a generator"""
    if isinstance(file, str):
        file_iter = open(file, 'rU')
    else:
        # If it's not a file name, assume it's a file object
        file_iter = file

    # for line_no, line in enumerate(file_iter):
    for line in file_iter:
        # print(line_no)
        line = line.strip().rstrip(',')                         # Remove trailing comma
        if ignore:
            if ordered:
                record = frozenset(
                    (index, value)
                    for index, value in enumerate(line.split(','))
                    if not ignore(index, value)
                )
            else:
                record = frozenset(value for value in line.split(',') if not ignore(None, value))
        else:
            if ordered:
                record = frozenset((index, value) for index, value in enumerate(line.split(',')))
            else:
                record = frozenset(line.split(','))
        yield record


if __name__ == "__main__":
    from optparse import OptionParser

    option_parser = OptionParser()
    option_parser.add_option('-f', '--inputFile',
                             dest='input',
                             help='filename containing csv',
                             default=None)
    option_parser.add_option('-s', '--minSupport',
                             dest='min_support',
                             help='minimum support value',
                             default=0.15,
                             type='float')
    option_parser.add_option('-c', '--minConfidence',
                             dest='min_confidence',
                             help='minimum confidence value',
                             default=0.6,
                             type='float')
    option_parser.add_option('-o', '--ordered',
                             action='store_true',
                             dest='ordered',
                             help='data consists of ordered columns',
                             default=False)
    option_parser.add_option('-i', '--ignore-nulls',
                             action='store_true',
                             dest='ignore_nulls',
                             help='ignore null values (blanks, "None", "NA")',
                             default=False)

    (options, args) = option_parser.parse_args()

    def null_value(value):
        value = value.strip().upper()
        return not value or value == 'NONE' or value == 'NA'

    if options.ignore_nulls:
        def value_filter(_, value):
            return null_value(value)
    else:
        value_filter = None

    if options.input is not None and not options.input:
        print('No data set filename specified, system with exit\n')
        sys.exit('System will exit')

    in_file = data_from_file(options.input or sys.stdin, options.ordered, value_filter)

    items, rules = run_apriori(in_file, options.min_support, options.min_confidence)

    print_results(items, rules)
