import logging
import sys
from datetime import datetime
from functools import wraps

LOGGER = logging.getLogger(__name__)


def log_handler(*loggers, logname=None):
    formatter = logging.Formatter(
        '%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # file handler
    if logname:
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

    for logger in loggers:
        if logname:
            logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)

def func_profile(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        cost_time = datetime.now() - start_time
        fullname = '{}.{}'.format(func.__module__, func.__name__)
        LOGGER.info('{}[kwargs={}] completed in {}'.format(
            fullname, kwargs, str(cost_time)
        ))
        return result
    return wrapped