from flask import jsonify
import logging
logger = logging.getLogger('digi_assistant')


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        logger.error(message)

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['result'] = self.message
        rv['status'] = self.status_code
        return rv


class DataNotFound(Exception):
    status_code = 501

    def __init__(self, message='', status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.error = 'Data not found'
        logger.error(self.error )

    def to_dict(self):
        rv = dict(self.payload or ())
        # rv['error'] = self.error
        rv['status'] = "error"
        rv['status_code'] = self.status_code
        rv['message'] = self.message
        return rv


class ElasticsearchTimeout(Exception):
    status_code = 503

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        logger.error(message)

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['result'] = self.message
        rv['status'] = self.status_code
        return rv


def invalid_usage_handler(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
