import logging
from time import time

def timer(func):
    def wrap(*args, **kwargs):
        starttime = time()
        result = func(*args, **kwargs)
        logging.info(f'Function {func.__name__!r} executed in {(time()-starttime):.4f}s')
        return result
    return wrap