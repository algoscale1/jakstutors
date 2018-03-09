from app import create_app
import logging
logging.basicConfig(level=logging.DEBUG)


app = create_app()


if __name__ == '__main__':
    logger = logging.getLogger('digi_assistant')
    logger.debug('****debug mode****')
    logger.info('digi_assistant scoring started')

    if app.debug:
        app.run(host="0.0.0.0", debug=True, port=8081, threaded=True)
    else:
        app.run(host="0.0.0.0", debug=False, port=8081, threaded=True)
