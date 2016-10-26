# coding: utf-8

"""
	author: Loïc M. DIVAD
	date: 2016-10-25
	see also: https://www.kaggle.com/c/digit-recognizer
"""

# ### imports
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from optparse import OptionParser
from sklearn.cross_validation import train_test_split

# ### constant
LOG_FORMAT = '%(asctime)s [ %(levelname)s ] : %(message)s'
LVL_MAP = {
 'debug'	:	logging.DEBUG, 
 'info'		:	logging.INFO, 
 'warn'		:	logging.WARN, 
 'error'	:	logging.ERROR, 
 'fatal'	:	logging.FATAL
}

# ### config
optparser = OptionParser()
optparser.add_option("-l", "--logginglvl", dest="logginglvl", default="info", help="Logging level")
optparser.add_option("-v", "--validation", dest="valid_set", default="0.20", help="Size of the validation set")
optparser.add_option("-f", "--fitupdates", dest="fit_infos", default="info", help="Whether or not compute fit updates")

options, args = optparser.parse_args()

# ### logging
logging.basicConfig(format=LOG_FORMAT)
log = logging.getLogger()
log.setLevel(LVL_MAP[options.logginglvl])


if __name__ == "__main__":

	log.debug("Starting the algorithms.")
	log.info("Starting the algorithms.")
	log.fatal("Starting the algorithms.")

