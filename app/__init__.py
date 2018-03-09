from flask import Flask
import logging
from logging.handlers import TimedRotatingFileHandler
from io import StringIO
import os
import configparser


debug = False
try:
    # A way to read the same config file using config-parser
    config_file = os.environ.get('DIGI_ASSISTANT_SETTINGS')
    config_file = os.path.join(os.path.dirname(__file__), config_file)
    config_str = "[section]\n" + open(config_file, 'r').read()
    configIO = StringIO(config_str)
    config = configparser.RawConfigParser()
    config.read_file(configIO)
    debug = config.get('section', 'DEBUG') == 'True'
except:
    pass

log_level = logging.DEBUG if debug else logging.ERROR
logger = logging.getLogger('digi_assistant')
file_handler = TimedRotatingFileHandler('logs/digi_assistant.log', 'D', 1, 10)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
logger.addHandler(file_handler)
logger.setLevel(log_level)


def create_app():
    app = Flask(__name__)
    app.config.from_object(__name__)
    app.config.from_envvar('DIGI_ASSISTANT_SETTINGS', silent=True)
    # app.config['DEBUG'] = True
    # app.debug = True

    from .routes import routes
    app.register_blueprint(routes)

    from .routes import exceptions
    app.register_error_handler(exceptions.InvalidUsage, exceptions.invalid_usage_handler)
    app.register_error_handler(exceptions.DataNotFound, exceptions.invalid_usage_handler)
    app.register_error_handler(exceptions.ElasticsearchTimeout, exceptions.invalid_usage_handler)
    return app