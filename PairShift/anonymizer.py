# The `anonymizer.py` is an helper script to anonymize raw data in the pair form. Use python3 to run the script: `python3 anonymizer.py -i raw_data.csv -o anonymized_data.csv`

import csv
from faker import Factory
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input-file",
    dest="input_file",
    help="read data from FILE",
    metavar="FILE")
parser.add_argument(
    "-o",
    "--output-file",
    dest="output_file",
    help="write data to FILE",
    metavar="FILE")

args = parser.parse_args()


def anonymize(dat):
    faker = Factory.create()
    names = defaultdict(faker.word)

    for row in dat:
        row['item1'] = names[row['item1']]
        row['item2'] = names[row['item2']]
        yield row


def convert(input_file, output_file):

    with open(input_file, 'rU') as i:
        reader = csv.DictReader(i)

        # anonymize data
        anonymized_data = anonymize(reader)

        with open(output_file, 'w') as o:
            writer = csv.DictWriter(o, reader.fieldnames)
            writer.writerow({k: k for k in reader.fieldnames})
            for row in anonymized_data:
                writer.writerow(row)


convert(args.input_file, args.output_file)
