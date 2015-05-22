"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATA_SET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATA_SET.csv -s 0.15 -c 0.6
"""

__author__ = 'Abhinav Saini'
__maintainer__ = 'Aaron Hosford'
__version__ = "0.1.0"

import csv
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
    for key, value in large_set.items():
        logger.info("Determining item support values for k = %s", key)
        result_items.extend([
            (tuple(item), get_support(item))
            for item in value
        ])

    result_rules = []
    for key, value in large_set.items():
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
                            (tuple(subset), tuple(remain)),
                            confidence)
                        )

    logger.info("Processing complete.")
    return result_items, result_rules


def itemset_to_string(itemset, ordered=False):
    """Converts an itemset to a readable string."""
    if ordered:
        return ', '.join(str(index) + ': ' + value for index, value in sorted(itemset))
    else:
        return ', '.join(value for value in sorted(itemset))


# noinspection PyShadowingNames
def print_itemsets(itemsets, ordered=False):
    """Prints the generated itemsets."""
    for itemset, support in sorted(itemsets, key=lambda pair: (pair[-1], pair), reverse=True):
        print("Itemset: %s  [%.3f]" % (itemset_to_string(itemset, ordered), support))


# noinspection PyShadowingNames
def print_rules(rules, ordered=False):
    """Prints the generated rules."""
    for (condition, prediction), confidence in sorted(rules, key=lambda pair: (pair[-1], pair), reverse=True):
        print("Rule: %s  =>  %s  [%.3f]" % (
            itemset_to_string(condition, ordered),
            itemset_to_string(prediction, ordered),
            confidence
        ))


# noinspection PyShadowingNames
def write_itemsets(path, itemsets, ordered=False, dialect='excel', *args, **kwargs):
    """
    Writes the itemsets out to a file in CSV format. The rows are organized as follows:
        "Support", "Size of Itemset", "Value1", ..., "ValueN"
    If the input file is ordered, indices are prepended to the values, with a separating colon, i.e. "Index: Value".
    """
    with open(path, 'w', newline='') as save_file:
        writer = csv.writer(save_file, dialect, *args, **kwargs)

        if ordered:
            for itemset, support in sorted(itemsets, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [support, len(itemset)]
                row.extend(str(index) + ': ' + value for index, value in sorted(itemset))
                writer.writerow(row)
        else:
            for itemset, support in sorted(itemsets, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [support, len(itemset)]
                row.extend(sorted(itemset))
                writer.writerow(row)


# noinspection PyShadowingNames
def write_rules(path, rules, ordered=False, dialect='excel', *args, **kwargs):
    """
    Writes the rules out to a file in CSV format. The rows are organized as follows:
        "Confidence", "Size of Condition", "Value1", ..., "ValueN", "=>", "Size of Prediction", "Value1", ..., "ValueM"
    If the input file is ordered, indices are prepended to the values, with a separating colon, i.e. "Index: Value".
    """
    with open(path, 'w', newline='') as save_file:
        writer = csv.writer(save_file, dialect, *args, **kwargs)

        if ordered:
            for (condition, prediction), confidence in sorted(rules, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [confidence, len(condition)]
                row.extend(str(index) + ': ' + value for index, value in sorted(condition))
                row.append('=>')
                row.append(len(prediction))
                row.extend(str(index) + ': ' + value for index, value in sorted(prediction))
                writer.writerow(row)
        else:
            for (condition, prediction), confidence in sorted(rules, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [confidence, len(condition)]
                row.extend(sorted(condition))
                row.append('=>')
                row.append(len(prediction))
                row.extend(sorted(prediction))
                writer.writerow(row)


# noinspection PyShadowingNames
def iter_rows(file_obj, ordered=False, ignore=None, dialect='excel', *args, **kwargs):
    """Return an iterator over the rows in the file."""
    reader = csv.reader(file_obj, dialect, *args, **kwargs)

    if ordered:
        if ignore:
            get_record = lambda row: frozenset(
                (index, value)
                for index, value in enumerate(row)
                if not ignore(index, value)
            )
        else:
            get_record = lambda row: frozenset((index, value) for index, value in enumerate(row))
    else:
        if ignore:
            get_record = lambda row: frozenset(value for index, value in enumerate(row) if not ignore(index, value))
        else:
            get_record = frozenset

    return (get_record(row) for row in reader if row and (len(row) > 1 or row[0]))


# noinspection PyShadowingNames
def data_from_file(file, ordered=False, ignore=None, dialect='excel', *args, **kwargs):
    """Function which reads from the file and returns a list of records"""
    if isinstance(file, str):
        with open(file, newline='') as file_iter:
            return list(iter_rows(file_iter, ordered, ignore, dialect, *args, **kwargs))
    else:
        # If it's not a file name, assume it's a file-like object
        return list(iter_rows(file, ordered, ignore, dialect, *args, **kwargs))


class FileIterator:
    """File iterator, for efficiently and repeatedly iterating over the records in a potentially large file without
    keeping it loaded in memory."""

    # noinspection PyShadowingNames
    def __init__(self, file_path, ordered=False, ignore=None, dialect='excel', *args, **kwargs):
        self.file_path = file_path
        self.ordered = bool(ordered)
        self.ignore = ignore
        self.dialect = dialect
        self.args = args
        self.kwargs = kwargs
        self._count = None

    def __len__(self):
        if self._count is None:
            counter = -1
            for counter, line in enumerate(open(self.file_path)):
                pass
            self._count = counter + 1
        return self._count

    def __iter__(self):
        with open(self.file_path, newline='') as file:
            for row in iter_rows(file, self.ordered, self.ignore, self.dialect, *self.args, **self.kwargs):
                yield row


if __name__ == "__main__":
    from optparse import OptionParser

    # TODO: Add command-line options to control log level, format, and path.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    option_parser = OptionParser()
    option_parser.add_option('-f', '--input-file',
                             dest='input',
                             help='filename containing csv',
                             default=None)
    option_parser.add_option('-s', '--min-support',
                             dest='min_support',
                             help='minimum support value',
                             default=0.15,
                             type='float')
    option_parser.add_option('-c', '--min-confidence',
                             dest='min_confidence',
                             help='minimum confidence value',
                             default=0.6,
                             type='float')
    option_parser.add_option('-o', '--ordered',
                             action='store_true',
                             dest='ordered',
                             help='data consists of ordered, indexable columns',
                             default=False)
    option_parser.add_option('-n', '--non-nulls',
                             action='store_true',
                             dest='non_nulls',
                             help='ignore null values (blanks, "None", "NA", "NULL")',
                             default=False)
    option_parser.add_option('-l', '--letters-only',
                             action='store_true',
                             dest='letters_only',
                             help='use only values that contain at least one letter of the alphabet',
                             default=False)
    option_parser.add_option('-e', '--exclude-columns',
                             dest='excluded',
                             help='exclude the comma-separated, zero-based column indices before processing',
                             default='',
                             type='string')
    option_parser.add_option('-m', '--in-memory',
                             action='store_true',
                             dest='in_memory',
                             help='load data to memory, rather than reading it from file repeatedly',
                             default=False)
    option_parser.add_option('-i', '--itemsets-file',
                             dest='itemsets',
                             help='filename where itemsets are saved (csv format)',
                             default=None)
    option_parser.add_option('-r', '--rules-file',
                             dest='rules',
                             help='filename where rules are saved (csv format)',
                             default=None)

    (options, args) = option_parser.parse_args()

    try:
        excluded_columns = {int(index) for index in options.excluded.split(',') if index}
    except ValueError:
        print("Badly-formed column index\n")
        sys.exit("System will exit")

    def null_value(value):
        """Returns True if the value is something which ought to be treated as a null."""
        value = value.strip().upper()
        return not value or value == 'NONE' or value == 'NA' or value == 'NULL'

    if options.letters_only:
        if options.non_nulls:
            if excluded_columns:
                def value_filter(index, value):
                    return index in excluded_columns or null_value(value) or not any(c.isalpha() for c in value)
            else:
                def value_filter(_, value):
                    return null_value(value) or not any(c.isalpha() for c in value)
        else:
            if excluded_columns:
                def value_filter(index, value):
                    return index in excluded_columns or not any(c.isalpha() for c in value)
            else:
                def value_filter(_, value):
                    return not any(c.isalpha() for c in value)
    else:
        if options.non_nulls:
            if excluded_columns:
                def value_filter(index, value):
                    return index in excluded_columns or null_value(value)
            else:
                def value_filter(_, value):
                    return null_value(value)
        else:
            if excluded_columns:
                def value_filter(index, _):
                    return index in excluded_columns
            else:
                value_filter = None

    if options.input is not None and not options.input:
        print('No data set filename specified, system with exit\n')
        sys.exit('System will exit')

    if options.in_memory or options.input is None:
        transactions_iterable = data_from_file(options.input or sys.stdin, options.ordered, value_filter)
    else:
        transactions_iterable = FileIterator(options.input, options.ordered, value_filter)

    itemsets, rules = run_apriori(transactions_iterable, options.min_support, options.min_confidence)

    if options.itemsets:
        write_itemsets(options.itemsets, itemsets, options.ordered)
    elif options.itemsets is None:
        print_itemsets(itemsets, options.ordered)

    if options.rules:
        write_rules(options.rules, rules, options.ordered)
    elif options.rules is None:
        print_rules(rules, options.ordered)
