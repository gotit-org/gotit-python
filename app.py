from flask import Flask, request, render_template, abort
from modules import face, object as obj, utility
from models import result


def create_app():
    # create and configure the app
    app = Flask("got it", instance_relative_config=True)
    # configure max size of payload content 
    app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

    # app routes here
    @app.before_request
    def check_auth():
        if not 'Authorization' in request.headers:
            abort(401)
        token = request.headers['Authorization']
        if not token:
            abort(401)

    @app.route("/", methods=["GET"])
    def index():
        return render_template('index.html')
    
    @app.route("/api/face/detect", methods=["POST"])
    def face_detection():
        return face.extract_face(request.files)
    
    @app.route("/api/face/match", methods=["POST"])
    def face_recognition():
        return face.recognition(request.json)
    
    @app.route("/api/object/detect", methods=["POST"])
    def object_detection():
        return obj.detect(request.files)

    @app.route("/api/object/match", methods=["POST"])
    def object_recognition():
        return obj.recognition(request.json)

    @app.errorhandler(400)
    def bad_request(error):
        return result.failed(None, str(error), 400)

    @app.errorhandler(401)
    def bad_request(error):
        return result.failed(None, str(error), 401)

    @app.errorhandler(404)
    def not_found(error):
        return result.failed(None, str(error), 404)

    @app.errorhandler(405)
    def method_not_allowed(error):
        return result.failed(None, str(error), 405)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return result.failed(None, str(error), 413)

    return app


if __name__ == "__main__":
    app = create_app()
    # For DEVELOPMENT only
    app.run(debug=False)
