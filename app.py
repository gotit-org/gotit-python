from flask import Flask, request, render_template
from modules import face, object as obj, utility


def create_app():
    # create and configure the app
    app = Flask("got it", instance_relative_config=True)
    # configure max size of payload content 
    app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024

    # app routes here
    @app.route("/")
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
    def object_match():
        return obj.match(request.json)

    return app


if __name__ == "__main__":
    app = create_app()
    # For DEVELOPMENT only
    app.run(debug=False)
