"""
Description: Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATA_SET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATA_SET.csv -s 0.15 -c 0.6
"""

# For full authorship and copyright information, see the mit-license file
__author__ = 'Aaron Hosford'
__version__ = "0.1.0"

import csv
import logging
import multiprocessing

from itertools import chain, combinations


def proper_subsets(item_set):
    """Return non-empty proper subsets of arr"""
    return chain(*(combinations(item_set, i) for i in range(1, len(item_set))))


def determine_support(item_sets, transactions, item_set_counts):
    for transaction in transactions:
        for item_set in item_sets:
            if item_set <= transaction:
                item_set_counts[item_set] = item_set_counts.get(item_set, 0) + 1



def get_items_with_min_support(item_sets, transactions, min_support, item_set_counts):
    """Calculates the support for items in the itemset and returns a subset
    of the itemset each of whose elements satisfies the minimum support."""
    determine_support(item_sets, transactions, item_set_counts)

    min_count = min_support * len(transactions)
    return [item_set for item_set in item_sets if item_set_counts.get(item_set, 0) >= min_count]


def join_sets(item_sets, length):
    """Join a set with itself and returns the n-element itemsets"""
    result = set()
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
        all_items |= transaction

    item_sets = []
    for item in all_items:
        item_sets.append(frozenset({item}))

    return item_sets


def run_apriori(transactions, min_support=.15, min_confidence=.6, sets=True, rules=True, get_candidates=join_sets, generational_max=None):
    """
    Run the apriori algorithm. transactions is a sequence of records which can be iterated over repeatedly,
    where each record is a set of items.

    Return both:
     - itemsets (tuple, support)
     - rules ((pre_tuple, post_tuple), confidence)
    """
    assert sets or rules

    logger = logging.getLogger(__name__)

    logger.info("Generating initial itemsets of size 1.")
    item_sets = get_initial_itemsets(transactions)

    item_set_counts = {}
    large_sets = [[]]

    logger.info("Identifying itemsets of size 1 with minimum support.")
    length_set = get_items_with_min_support(
        item_sets,
        transactions,
        min_support,
        item_set_counts
    )

    size = 1
    while length_set:
        print("size:", size)
        print("count:", len(length_set))

        if generational_max and len(length_set) > generational_max:
            length_set.sort(key=item_set_counts.get, reverse=True)
            length_set = length_set[:generational_max]

        large_sets.append(length_set)
        size += 1

        logger.info("Generating itemsets of size %s.", size)
        candidate_set = get_candidates(length_set, size)

        logger.info("Identifying itemsets of size %s with minimum support.", size)
        length_set = get_items_with_min_support(
            candidate_set,
            transactions,
            min_support,
            item_set_counts
        )

    if sets:
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

    if rules:
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
                        if subset not in item_set_counts:
                            determine_support([subset], transactions, item_set_counts)
                        confidence = item_set_counts[item_set] / item_set_counts[subset]
                        if confidence >= min_confidence:
                            result_rules.append((
                                (subset, remain),
                                confidence)
                            )

    logger.info("Processing complete.")

    if sets:
        if rules:
            return result_items, result_rules
        return result_items
    else:
        return result_rules


# IDEA:
#   - Partition the transactions into disjoint subsets.
#   - Multiple processes, local or on other machines, each running Apriori on one of the subsets.
#   - Each process is wrapped in a manager object of some sort that conveys data and watches the
#     process to determine when it is complete, hiding implementation details so we don't have to
#     worry about whether the process is local or not.
#   - The Merge object waits on processes to complete. Then it runs the Apriori algorithm on the
#     full data set, but instead of simply joining item sets as in join_sets(), the Merge object
#     also filters out any item set that doesn't appear in the results from at least one of the
#     subsets. In other words, it pre-filters the candidate set. This should work reliably
#     because in order for the support of an item set within a subset to drop below the minimum
#     density, those records must be moved to another subset, pushing its density above the
#     threshold.
class AprioriMerge:

    def __init__(self, candidates):
        self.candidates = frozenset(candidates)

    def get_candidates(self, length_set, size):
        """Finds each pair of sets in length set for which the union is one of the previously identified candidates."""
        return {item_set for item_set in join_sets(length_set, size) if item_set in self.candidates}

    def __call__(self, transactions, min_support=.15, min_confidence=.6, sets=True, rules=True):
        return run_apriori(transactions, min_support, min_confidence, sets, rules, get_candidates=self.get_candidates)


def run_distributed_apriori(transactions, min_support=.15, min_confidence=.6, sets=True, rules=True, generational_max=None, chunk_size=5000, dispatcher=None):
    if dispatcher is None:
        dispatcher = LocalDispatcher(multiprocessing.cpu_count())
    chunk = []
    candidates = set()
    for index, transaction in enumerate(transactions):
        if chunk and not index % chunk_size:
            print("Dispatching chunk of length", chunk_size)
            dispatcher.dispatch(chunk, min_support, min_confidence, rules=False, generational_max=generational_max)
            chunk = []
        for results in dispatcher.results():
            print("Processing results of length", len(results))
            candidates.update(results)
        chunk.append(transaction)
    if chunk:
        print("Dispatching chunk of length", chunk_size)
        dispatcher.dispatch(chunk, min_support, min_confidence, rules=False, generational_max=generational_max)
    while dispatcher.more():
        dispatcher.wait()
        for results in dispatcher.results():
            print("Processing results of length", len(results))
            candidates.update(results)
    print("Total candidates identified:", len(candidates))
    merger = AprioriMerge(candidates)
    return merger(transactions, min_support, min_confidence, sets, rules)


def _execute(results_queue, args, kwargs):
    results_queue.put(run_apriori(*args, **kwargs))


class LocalDispatcher:

    def __init__(self, max_running=None):
        self._manager = multiprocessing.Manager()
        self._results_queue = self._manager.Queue()
        self._pool = multiprocessing.Pool()
        self._running = []
        self._counter = 0
        self._max_running = max_running

    def dispatch(self, *args, **kwargs):
        if self._max_running:
            while len(self._running) >= self._max_running:
                self.wait()
        self._counter += 1
        self._running.append(self._pool.apply_async(_execute, (self._results_queue, args, kwargs)))

    def wait(self):
        for async_result in self._running:
            if async_result.ready():
                async_result.get(0)
                self._running.remove(async_result)
                break
            else:
                async_result.wait(1)

    def results(self):
        while not self._results_queue.empty():
            self._counter -= 1
            result = self._results_queue.get()
            yield result

    def more(self):
        return self._counter > 0


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
