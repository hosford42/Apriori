Python Implementation of Apriori Algorithm 
==========================================


Command-Line Usage
------------------
To run the program with dataset provided and default values for _min_support_(0.15) and *min_confidence*(0.6)

    python apriori.py -f INTEGRATED-DATASET.csv

To run program with dataset  

    python apriori.py -f INTEGRATED-DATASET.csv -s 0.17 -c 0.68

Best results are obtained for the following values of support and confidence:  

Support     : Between 0.1 and 0.2  

Confidence  : Between 0.5 and 0.7


Python Interface
----------------

The primary entrypoint to the package is the `run_apriori()` method. You will need to load your data into an iterable
object, either using the `data_from_file()` function or the `FileIterator()` class, and then pass this to
`run_apriori()` along with a minimum support and a minimum confidence. The `data_from_file()` function loads all the
data into memory at once, which is fast for smaller files but can cause problems for larger ones. The `FileIterator()`
class keeps the data on disk and loads only what is needed as it is needed, which is slow for smaller data sets but
avoids memory issues that can slow the process down for larger ones.

If the records you are processing is columnar, where the changing the order of the values in a row affects its meaning,
be sure to set the `ordered` flag when loading your data with `data_from_file()` or `FileIterator()`. This will convert
each value in the row to an `(index, value)` pair so the algorithm treats the values as non-reorderable. By default,
the algorithm assumes the order of the values in a row is inconsequential.

`run_apriori()` returns a pair, `(itemsets, rules)`. The `write_itemsets()` and `write_rules()` functions are provided
as a convenience for saving the results to file.


Example Usage:

    from apriori import run_apriori, data_from_file, FileIterator

    data_csv_path = 'your_source_data.csv'
    itemsets_csv_path = 'itemsets_save_loc.csv'
    rules_csv_path = 'rules_save_loc.csv'

    dataset_is_small = True  # This depends on your hardware
    data_is_columnar = True  # Whether the order of the values appearing in a given row matters

    min_support = .15
    min_confidence = .6

    if dataset_is_small:
        data_iterable = data_from_file(data_csv_path, ordered=data_is_columnar)
    else:
        data_iterable = FileIterator(data_csv_path, ordered=data_is_columnar)

    itemsets, rules = run_apriori(data_iterable, min_support, min_confidence)

    write_itemsets(itemsets_csv_path, itemsets, ordered=data_is_columnar)
    write_rules(rules_csv_path, rules, ordered=data_is_columnar)



License
-------
MIT-License


Notes
-----

INTEGRATED-DATASET.csv is a copy of the “Online directory of certified businesses with a detailed profile” file from the
Small Business Services (SBS) dataset in the `NYC Open Data Sets <http://nycopendata.socrata.com/>`_

-------
