import logging

logger = logging.getLogger('data_gen')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:  %(funcName)s:  %(levelname)s:  %(message)s')
file_handler = logging.FileHandler('logs\\data_gen.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
