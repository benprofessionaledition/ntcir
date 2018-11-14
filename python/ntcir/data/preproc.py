"""
Preprocesses a bunch of json shit
Usage:
    preproc.py <filename>
"""
import json
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__, version=1.)
    tgt_file = args["<filename>"]
    with open(tgt_file) as f:
        tgt_json = json.load(f)
