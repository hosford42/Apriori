"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATA_SET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATA_SET.csv -s 0.15 -c 0.6
"""

import logging
import sys

from itertools import chain, combinations
from collections import defaultdict


def subsets(arr):
    """Return non-empty subsets of arr"""
    return chain(*(combinations(arr, i + 1) for i in range(len(arr))))


def get_items_with_min_support(item_set, transactions, min_support, freq_set):
    """Calculates the support for items in the itemset and returns a subset
    of the itemset each of whose elements satisfies the minimum support."""
    results = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transactions:
            if item <= transaction:
                freq_set[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = count / len(transactions)

        if support >= min_support:
            results.add(item)

    return results


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return {i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length}


def get_initial_itemsets(transactions):
    item_set = set()
    for transaction in transactions:
        for item in transaction:
            item_set.add(frozenset([item]))              # Generate 1-itemsets
    return item_set


def run_apriori(data_iter, min_support, min_confidence):
    """
    Run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pre_tuple, post_tuple), confidence)
    """
    logger = logging.getLogger(__name__)

    logger.info("Getting initial itemsets")
    item_set = get_initial_itemsets(data_iter)

    freq_set = defaultdict(int)
    large_set = {}
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy min_support

    # Dictionary which stores Association Rules

    logger.info("Getting items with minimum support for k = 1")
    current_length_set = get_items_with_min_support(
        item_set,
        data_iter,
        min_support,
        freq_set
    )

    k = 2
    while current_length_set:
        large_set[k-1] = current_length_set
        logger.info("Joining subsets for k = %s", k)
        current_length_set = join_set(current_length_set, k)
        logger.info("Getting items with minimum support for k = %s", k)
        current_candidate_set = get_items_with_min_support(
            current_length_set,
            data_iter,
            min_support,
            freq_set
        )
        current_length_set = current_candidate_set
        k += 1

    # noinspection PyShadowingNames
    def get_support(item):
        """Local function which returns the support of an item"""
        return freq_set[item] / len(data_iter)

    result_items = []
    for key, value in sorted(large_set.items()):
        logger.info("Determining item support values for k = %s", key)
        result_items.extend([
            (tuple(item), get_support(item))
            for item in value
        ])

    result_rules = []
    for key, value in sorted(large_set.items()):
        if key < 2:
            continue
        logger.info("Rule confidence values for k = %s", key)
        for item in value:
            for subset in subsets(item):
                subset = frozenset(subset)
                remain = item.difference(subset)
                if len(remain) > 0:
                    confidence = get_support(item) / get_support(subset)
                    if confidence >= min_confidence:
                        result_rules.append((
                            (tuple(sorted(subset)), tuple(sorted(remain))),
                            confidence)
                        )

    logger.info("Processing complete.")
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


class FileIterator:
    """File iterator, for efficiently and repeatedly iterating over the records in a potentially large file without
    keeping it loaded in memory."""

    def __init__(self, file_path, ordered=False, ignore=None):
        self.file_path = file_path
        self.ordered = bool(ordered)
        self.ignore = ignore
        self._count = None

    @staticmethod
    def _get_simple_record(line):
        line = line.strip().rstrip(',')                         # Remove trailing comma
        return frozenset(line.split(','))

    def _get_filtered_record(self, line):
        line = line.strip().rstrip(',')                         # Remove trailing comma
        return frozenset(value for value in line.split(',') if not self.ignore(None, value))

    @staticmethod
    def _get_ordered_record(line):
        line = line.strip().rstrip(',')                         # Remove trailing comma
        return frozenset((index, value) for index, value in enumerate(line.split(',')))

    def _get_ordered_filtered_record(self, line):
        line = line.strip().rstrip(',')                         # Remove trailing comma
        return frozenset(
            (index, value)
            for index, value in enumerate(line.split(','))
            if not self.ignore(index, value)
        )

    def __len__(self):
        if self._count is None:
            counter = -1
            for counter, line in open(self.file_path):
                pass
            self._count = counter + 1
        return self._count

    def __iter__(self):
        # By making the choice here, it is avoided once per line later on, which should give a slight speed boost
        if self.ordered:
            if self.ignore:
                get_record = self._get_ordered_filtered_record
            else:
                get_record = self._get_ordered_record
        else:
            if self.ignore:
                get_record = self._get_filtered_record
            else:
                get_record = self._get_simple_record

        if self._count is None:
            with open(self.file_path) as file:
                counter = -1
                for counter, line in enumerate(file):
                    yield get_record(line)
                self._count = counter + 1
        else:
            with open(self.file_path) as file:
                for line in file:
                    yield get_record(line)


if __name__ == "__main__":
    from optparse import OptionParser

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

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
        """Returns True if the value is something which ought to be treated as a null."""
        value = value.strip().upper()
        return not value or value == 'NONE' or value == 'NA' or value == 'NULL'

    if options.ignore_nulls:
        def value_filter(_, value):
            return null_value(value)
    else:
        value_filter = None

    if not options.input:
        print('No data set filename specified, system with exit\n')
        sys.exit('System will exit')

    file_iterator = FileIterator(options.input, options.ordered, value_filter)
    items, rules = run_apriori(file_iterator, options.min_support, options.min_confidence)

    print_results(items, rules)
