import os
from flask import request, jsonify
from . import routes
from flask import render_template


@routes.route('/login')
def get_login():
    return render_template('login.html')


