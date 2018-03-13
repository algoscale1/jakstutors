from flask import Blueprint,render_template
routes = Blueprint('routes',  __name__)


@routes.route('/index')
def index():
    print ("testtttttttt")
    return render_template('index.html')
