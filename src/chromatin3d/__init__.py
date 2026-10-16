import logging

# create logger
logger = logging.getLogger("chromatin3d")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
fh = logging.FileHandler("chromatin3d.log")
fh.setLevel(logging.DEBUG)

# log_format = "%(asctime)s | %(name)s | %(levelname)s | %(pathname)s:%(lineno)d | %(message)s"
log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
date_format = "%Y.%m.%d %H:%M:%S"

# create formatter
formatter = logging.Formatter(log_format, datefmt=date_format)

# add formatter to ch
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(fh)
