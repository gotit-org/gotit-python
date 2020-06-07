from flask import jsonify, make_response

def success(data, message="Process done successfuly", status_code=200):
    return make_response(jsonify({
        "isSucceeded": True,
        "data": data,
        "message": message
    }), status_code)

def failed(data, message="Process Failed!", status_code=500):
    return make_response(jsonify({
        "isSucceeded": False,
        "data": data,
        "message": message
    }), status_code)