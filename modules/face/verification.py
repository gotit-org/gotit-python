# face verification with the VGGFace2 model
from modules.singleton import mtcnn, vgg, graph
from numpy import asarray, random, fromstring, float32, reshape
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from PIL import Image
from models.response_models import FaceDetected, MatchScore, FaceVerificationResult
from models import result
from modules import utility
import base64
import cv2


def draw_face_box(image, face_box):
    x, y, w, h = face_box[:]
    cv2.rectangle(image, (x, y), (x + w, y + h),
        random.uniform(low=0, high=255, size=(3)), 2)


def extract_face(images) -> list:
    try:
        detector = mtcnn.get_instace()
        resultData = list()
        imagesData = utility.read_images(images)
        
        with graph.get_instace().as_default():
            for data in imagesData:
                detectedFaces = detector.detect_faces(data["image"])
                faces = list()
                for face in detectedFaces:
                    if face['confidence'] > 0.95:
                        face['box'][0] = max(0, face['box'][0])
                        face['box'][1] = max(0, face['box'][1])
                        face['confidence'] = round(face['confidence'], 2)
                        faces.append(face)
                
                if len(faces) > 0 :
                    resultData.append(FaceDetected(
                        data["name"], len(faces), faces).__dict__)

        return result.success(resultData, " Can't detect any face!" if len(resultData) == 0 else "Process done successfuly")

    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(known_data):
    faces = list()
    for data in known_data:
        # crop face from image
        data['image'] = utility.read_b64image(data['image'])
        x1, y1 = data['box'][0], data['box'][1]
        x2, y2 = x1 + data['box'][2], y1 + data['box'][3]
        face = data['image'][y1:y2, x1:x2]

        # resize pixels to the model size
        face = Image.fromarray(face)
        face = face.resize((224, 224))
        faces.append(asarray(face))

    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    model = vgg.get_instace()
    # perform prediction
    with graph.get_instace().as_default():
        yhat = model.predict(samples)

    return yhat


def match(known_embedding, candidate_embedding):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    return score


def recognition(json_data):
    try:
        MATCH_THRESHOLD = 0.5
        knowns = get_embeddings(json_data['known'])
        candidates = json_data['candidates']
        score_list = list()
        for candidate in candidates:
            embeddenges = decode_embeddings(candidate['embedding'])
            min_score = None
            for embeddenge in embeddenges:
                for known in knowns:
                    score = match(known, embeddenge)
                    if score <= MATCH_THRESHOLD and (min_score == None or score < min_score):
                        min_score = score
                
            # append min score to result list if and only if it's less than threshold
            if min_score != None:
                score_list.append(MatchScore(
                    candidate['id'], min_score).__dict__)

        score_list.sort(key=lambda x: x['score'])
        return result.success(FaceVerificationResult(score_list, base64.b64encode(knowns).decode('utf-8')).__dict__, 
            'no matches found!' if len(score_list) == 0 else 'Matches!!!')

    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def decode_embeddings(encode_embeddings):
    TARGET_SIZE = 2048
    decoded = fromstring(base64.b64decode(encode_embeddings), float32)
    n = int(decoded.shape[0] / TARGET_SIZE)
    return reshape(decoded, (n, TARGET_SIZE))
