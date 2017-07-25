"""
Description: Python implementation of the Apriori Algorithm
=======
An implementation of the Apriori algorithm for Python 3.
"""

import csv
import logging

from collections import defaultdict
from itertools import combinations, chain

from hashable_arrays import BitArray


# For full authorship and copyright information, see the mit-license file
__author__ = 'Abhinav Saini, Aaron Hosford'
__version__ = "0.1"


def proper_subsets(item_set):
    """Return non-empty proper subsets of arr"""
    return chain(*(combinations(item_set, i) for i in range(1, len(item_set))))


# TODO: I think I need to switch from ints to actual bitarrays of some sort. It's taking much longer to run like this.


def subsets(item_set):
    """Return an iterator over all subsets of the item set."""
    indices = [index for index in range(item_set.bit_length()) if item_set & (1 << index)]
    for combination in chain(*(combinations(indices, size) for size in range(1, len(indices) + 1))):
        result = 0
        for index in combination:
            result |= (1 << index)
        yield result


def get_initial_item_sets(bit_count):
    """Return the item sets of size 1."""
    item_sets = []
    bit_list = [False] * bit_count
    for index in range(bit_count):
        bit_list[index] = True
        item_sets.append(BitArray(bit_list))
        bit_list[index] = False
    return item_sets


def join_sets(item_sets, length):
    """Combine the item sets together, returning the unique combinations that have the requested length."""
    result = set()
    for index1, item_set1 in enumerate(item_sets):
        for index2 in range(index1 + 1, len(item_sets)):
            item_set2 = item_sets[index2]
            combined = item_set1 | item_set2
            if combined.bit_count() == length:
                result.add(combined)
    return result


def get_items_with_min_support(item_sets, transactions_bitmasks, min_support, item_set_counts):
    """Determine the occurrence count for each item set. Filter out the item sets that have insufficient support,
    returning those with sufficient support as a list."""

    for transaction in transactions_bitmasks:
        inv_transaction = ~transaction
        for item_set in item_sets:
            if not (item_set & inv_transaction):
                item_set_counts[item_set] += 1

    min_count = min_support * len(transactions_bitmasks)
    return [item_set for item_set in item_sets if item_set_counts[item_set] >= min_count]


def run_apriori(transactions, min_support=.15, min_confidence=.6):
    """
    Run the Apriori algorithm, returning a pair (item_sets, rules), where item_sets is a list of pairs
    (item_set, support), and rules is a list of pairs ((condition, prediction), confidence). The transactions argument
    should be a sequence of records which can be iterated over repeatedly, where each record is a set of items.
    """

    logger = logging.getLogger(__name__)

    logger.info("Converting transactions to bitmasks.")
    converter = BitConverter.from_transactions(transactions)
    transaction_bitmasks = [converter.to_bits(transaction) for transaction in transactions]

    logger.info("Generating initial item sets of size 1.")
    item_sets = get_initial_item_sets(converter.bit_count)

    item_set_counts = defaultdict(int)
    large_sets = [[]]

    logger.info("Identifying item sets of size 1 with minimum support.")
    current_length_set = get_items_with_min_support(
        item_sets,
        transaction_bitmasks,
        min_support,
        item_set_counts
    )

    size = 1
    while current_length_set:
        large_sets.append(current_length_set)
        size += 1

        logger.info("Generating item sets of size %s.", size)
        current_length_set = join_sets(current_length_set, size)

        logger.info("Identifying item sets of size %s with minimum support.", size)
        current_candidate_set = get_items_with_min_support(
            current_length_set,
            transaction_bitmasks,
            min_support,
            item_set_counts
        )

        # Free up some wasted space.
        for item_set in current_length_set:
            if item_set not in current_candidate_set:
                del item_set_counts[item_set]

        current_length_set = current_candidate_set

    result_items = []
    transaction_count = len(transaction_bitmasks)
    for size, item_sets in enumerate(large_sets):
        if size < 1:
            continue
        logger.info("Determining support values for item sets of size %s.", size)
        # support = (# of occurrences) / (total # of transactions)
        result_items.extend(
            (converter.from_bits(item_set), item_set_counts[item_set] / transaction_count)
            for item_set in item_sets
        )

    result_rules = []
    for size, item_sets in enumerate(large_sets):
        if size < 2:
            continue
        logger.info("Determining rule confidence values for item sets of size %s.", size)
        for item_set in item_sets:
            for subset in subsets(item_set):
                if not subset or subset == item_set:
                    continue
                remain = item_set - subset
                if remain.bit_length() > 0:
                    # support = (# of occurrences) / (total # of transactions)
                    # confidence = (support for item_set) / (support for subset)
                    confidence = item_set_counts[item_set] / item_set_counts[subset]
                    if confidence >= min_confidence:
                        result_rules.append((
                            (converter.from_bits(subset), converter.from_bits(remain)),
                            confidence)
                        )

    logger.info("Processing complete.")
    return result_items, result_rules


def item_set_to_string(item_set, ordered=False):
    """Converts an item set to a readable string."""
    if ordered:
        return ', '.join(str(index) + ': ' + value for index, value in sorted(item_set))
    else:
        return ', '.join(str(value) for value in sorted(item_set))



def print_item_sets(item_sets, ordered=False):
    """Prints the generated item sets."""
    for item_set, support in sorted(item_sets, key=lambda pair: (pair[-1], pair), reverse=True):
        print("Item set: %s  [%.3f]" % (item_set_to_string(item_set, ordered), support))


def print_rules(rules, ordered=False):
    """Prints the generated rules."""
    for (condition, prediction), confidence in sorted(rules, key=lambda pair: (pair[-1], pair), reverse=True):
        print("Rule: %s  =>  %s  [%.3f]" % (
            item_set_to_string(condition, ordered),
            item_set_to_string(prediction, ordered),
            confidence
        ))


# noinspection PyShadowingNames
def write_item_sets(path, item_sets, ordered=False, dialect='excel', *args, **kwargs):
    """
    Writes the item_sets out to a file in CSV format. The rows are organized as follows:
        "Support", "Size of Item Set", "Value1", ..., "ValueN"
    If the input file is ordered, indices are prepended to the values, with a separating colon, i.e. "Index: Value".
    """
    with open(path, 'w', newline='') as save_file:
        writer = csv.writer(save_file, dialect, *args, **kwargs)

        if ordered:
            for item_set, support in sorted(item_sets, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [support, len(item_set)]
                row.extend(str(index) + ': ' + value for index, value in sorted(item_set))
                writer.writerow(row)
        else:
            for item_set, support in sorted(item_sets, key=lambda pair: (pair[-1], pair), reverse=True):
                row = [support, len(item_set)]
                row.extend(sorted(item_set))
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
            def get_record(row):
                return frozenset(
                    (index, value)
                    for index, value in enumerate(row)
                    if not ignore(index, value)
                )
        else:
            def get_record(row):
                return frozenset((index, value) for index, value in enumerate(row))
    else:
        if ignore:
            def get_record(row):
                return frozenset(value for index, value in enumerate(row) if not ignore(index, value))
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


# TODO: Check out the bitsets library, available via pip. It does pretty much the same thing.
class BitConverter:
    """Converts item sets to integer bitmasks and vice versa."""

    @classmethod
    def _iter_transaction_items(cls, transactions):
        for transaction in transactions:
            for item in transaction:
                yield item

    @classmethod
    def from_transactions(cls, transactions, frozen=True):
        """Create a BitConverter from a sequence of transactions. This eliminates the need to construct a single set
        or list of values before passing it in to the constructor."""
        return cls(cls._iter_transaction_items(transactions), frozen)

    def __init__(self, items, frozen=True):
        self._items = []
        self._item_index_map = {}
        self.frozen = frozen

        if isinstance(items, (set, frozenset)):
            self._items.extend(items)
            del items  # Don't keep it around any longer than necessary, as it may be very large.
            for index, item in enumerate(self._items):
                self._item_index_map[item] = index
        else:
            for item in items:
                if item not in self._item_index_map:
                    self._item_index_map[item] = len(self._items)
                    self._items.append(item)

        self._bit_count = len(self._items)

    @property
    def bit_count(self):
        """The number of bits used to represent the sets."""
        return self._bit_count

    def to_bits(self, item_set):
        """Return an integer that represents the given item set."""
        return BitArray.from_indices((self._item_index_map[item] for item in item_set), self._bit_count)

    def from_bits(self, bits):
        """Return the item set that the given integer represents."""
        result = set()
        for index, bit in enumerate(bits):
            if bit:
                result.add(self._items[index])

        if self.frozen:
            return frozenset(result)
        else:
            return result


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
    option_parser.add_option('-i', '--item-sets-file',
                             dest='item_sets',
                             help='filename where item sets are saved (csv format)',
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

    if options.input is None:
        transactions_iterable = data_from_file(sys.stdin, options.ordered, value_filter)
    else:
        transactions_iterable = FileIterator(options.input, options.ordered, value_filter)

    item_sets, rules = run_apriori(transactions_iterable, options.min_support, options.min_confidence)

    if options.item_sets:
        write_item_sets(options.item_sets, item_sets, options.ordered)
    elif options.item_sets is None:
        print_item_sets(item_sets, options.ordered)

    if options.rules:
        write_rules(options.rules, rules, options.ordered)
    elif options.rules is None:
        print_rules(rules, options.ordered)
