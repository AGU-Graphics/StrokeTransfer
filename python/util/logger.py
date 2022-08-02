import logging

LEVEL = logging.INFO


def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s: %(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger
