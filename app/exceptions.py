import logging

logger = logging.getLogger('jakstutors')


class DataNotFoundError(Exception):

    def __init__(self, message, error=None):
        Exception.__init__(self)
        self.message = message
        logger.error('DataNotFound: ' + message)
        if error:
            logger.error(str(error))


class InvalidData(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message
        logger.error('InvalidData: ' + message)


class ElasticsearchService(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message
        logger.error('ElasticsearchService: ' + message)
