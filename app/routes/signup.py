import os
from flask import request, jsonify
from . import routes
from flask import render_template


@routes.route('/signup')
def get_signup():
    return render_template('signup.html')





