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
import multiprocessing

from itertools import chain, combinations, cycle
from collections import defaultdict


def proper_subsets(item_set):
    """Return non-empty proper subsets of arr"""
    return chain(*(combinations(item_set, i) for i in range(1, len(item_set))))


def issubset(item_set_transaction):
    item_set, transaction = item_set_transaction
    return item_set if item_set.issubset(transaction) else None
    # return item_set <= transaction

def get_items_with_min_support(item_sets, transactions, min_support, item_set_counts, pool=None):
    """Calculates the support for items in the itemset and returns a subset
    of the itemset each of whose elements satisfies the minimum support."""

    if pool:
        for transaction in transactions:
            for item_set in pool.imap_unordered(issubset, zip(item_sets, cycle([transaction]))):
                if item_set is not None:
                    item_set_counts[item_set] += 1
    else:
        for transaction in transactions:
            for item_set in item_sets:
                if item_set <= transaction:
                    item_set_counts[item_set] += 1

    min_count = min_support * len(transactions)
    return [item_set for item_set in item_sets if item_set_counts[item_set] >= min_count]


def join_pair(i_j_length):
    i, j, length = i_j_length
    u = i | j
    if len(u) == length:
        return u
    else:
        return None

def join_sets(item_sets, length, pool=None):
    """Join a set with itself and returns the n-element itemsets"""
    result = set()
    if pool:
        for i in item_sets:
            for u in pool.imap_unordered(join_pair, zip(cycle([i]), item_sets, cycle([length]))):
                if u is not None:
                    result.add(u)
    else:
        for i in item_sets:
            for j in item_sets:
                u = i | j
                if len(u) == length:
                    result.add(u)
    return result


def get_initial_itemsets(transactions):
    """Return the itemsets of size 1."""
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)

    item_sets = []
    for item in all_items:
        item_sets.append(frozenset({item}))

    return item_sets


def run_apriori(transactions, min_support=.15, min_confidence=.6, use_pool=True):
    """
    Run the apriori algorithm. transactions is a sequence of records which can be iterated over repeatedly,
    where each record is a set of items.

    Return both:
     - itemsets (tuple, support)
     - rules ((pre_tuple, post_tuple), confidence)
    """
    logger = logging.getLogger(__name__)

    if use_pool:
        logger.info("Using process pool.")
        pool = multiprocessing.Pool()
    else:
        pool = None

    logger.info("Generating initial itemsets of size 1.")
    item_sets = get_initial_itemsets(transactions)

    item_set_counts = defaultdict(int)
    large_sets = [[]]

    logger.info("Identifying itemsets of size 1 with minimum support.")
    current_length_set = get_items_with_min_support(
        item_sets,
        transactions,
        min_support,
        item_set_counts,
        pool
    )

    size = 1
    while current_length_set:
        large_sets.append(current_length_set)
        size += 1

        logger.info("Generating itemsets of size %s.", size)
        current_length_set = join_sets(current_length_set, size)

        logger.info("Identifying itemsets of size %s with minimum support.", size)
        current_candidate_set = get_items_with_min_support(
            current_length_set,
            transactions,
            min_support,
            item_set_counts,
            pool
        )

        current_length_set = current_candidate_set

    result_items = []
    transaction_count = len(transactions)
    for size, item_sets in enumerate(large_sets):
        if not size:
            continue
        logger.info("Determining support values for itemsets of size %s.", size)
        # support = (# of occurrences) / (total # of transactions)
        result_items.extend(
            (item_set, item_set_counts[item_set] / transaction_count)
            for item_set in item_sets
        )

    result_rules = []
    for size, item_sets in enumerate(large_sets):
        if size < 2:
            continue
        logger.info("Determining rule confidence values for itemsets of size %s.", size)
        for item_set in item_sets:
            for subset in proper_subsets(item_set):
                subset = frozenset(subset)
                remain = frozenset(item_set.difference(subset))
                if len(remain) > 0:
                    # support = (# of occurrences) / (total # of transactions)
                    # confidence = (support for item_set) / (support for subset)
                    confidence = item_set_counts[item_set] / item_set_counts[subset]
                    if confidence >= min_confidence:
                        result_rules.append((
                            (subset, remain),
                            confidence)
                        )

    logger.info("Processing complete.")
    return result_items, result_rules


def itemset_to_string(itemset, ordered=False):
    """Converts an itemset to a readable string."""
    if ordered:
        return ', '.join(str(index) + ': ' + value for index, value in sorted(itemset))
    else:
        return ', '.join(str(value) for value in sorted(itemset))


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
                row.extend(str(index) + ': ' + str(value) for index, value in sorted(itemset))
                writer.writerow(row)
        else:
            for itemset, support in sorted(itemsets, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [support, len(itemset)]
                row.extend(str(item) for item in sorted(itemset))
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
                row.extend(str(index) + ': ' + str(value) for index, value in sorted(condition))
                row.append('=>')
                row.append(len(prediction))
                row.extend(str(index) + ': ' + str(value) for index, value in sorted(prediction))
                writer.writerow(row)
        else:
            for (condition, prediction), confidence in sorted(rules, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [confidence, len(condition)]
                row.extend(str(item) for item in sorted(condition))
                row.append('=>')
                row.append(len(prediction))
                row.extend(str(item) for item in sorted(prediction))
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
    import sys

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
