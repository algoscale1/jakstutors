import os
from flask import request, jsonify
from . import routes
from .exceptions import InvalidUsage, DataNotFound, ElasticsearchTimeout
from app.exceptions import InvalidData,  ElasticsearchService, DataNotFoundError
import logging
# from app.base import Base
from jaks_model import process_n_get_text


@routes.route('/classify', methods=['POST'])
def get_classification():
    """

    :return:
    """
    print(request,"*******")
    ff = request.files
    print("HHHHHHHH")
    print(ff,"FILE HEREEEEEEEEE")   
    if len(ff)==0:
	
        return "{'status':200,'message':'No file was sent'}"
    if not os.path.exists("test"):
        os.makedirs("test")
        print("DIRECTORY CREATED")
    for fi in ff:
        print("FILE SAVING")
        ff[fi].save('test/'+ff[fi].filename)


    extracted_data = process_n_get_text()


    return jsonify(extracted_data)
