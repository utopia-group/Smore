"""
This file stores some general config values for scripts, but not specific
to running tasks as those are currently stored in run_config.py.
"""
import random

SEED = 66
random.seed(SEED)


AE = True


# Credentials for Google Sheet API
credentials = {}


CACHE_PATH = '.cache'
CACHE_IN_MEMORY = True
IN_MEMORY_CACHE_SIZE = 500000


def set_cache_path(new_cache_path):
    global CACHE_PATH
    CACHE_PATH = new_cache_path


PIPELINE_DEBUG = False


def pd_print(*args):
    global PIPELINE_DEBUG
    if PIPELINE_DEBUG:
        print(' '.join([repr(s) for s in args]))
