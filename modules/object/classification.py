from modules.singleton import mrcnn, graph
from modules import utility
from models import result
from models.response_models import DetectionResult, DetectedBox, MatchScore, MatchResult
from setuptools import sandbox
from scipy.spatial.distance import cosine
import numpy as np
import os
import base64
try:
    from . import ColorPy
except ImportError:
    # setup ColorPy
    SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
    sandbox.run_setup(SETUP_DIR + '/setup.py', ['build_ext', '--inplace'])
    from . import ColorPy


def detect(images):
    try:
        # init variables
        flag = False
        TARGET_SIZE = (448, 448)
        CONFIDENCE_THRESHOLD = 0.60

        # init result list
        match_result = list()

        # load model & classes
        model = mrcnn.get_model()
        classes = mrcnn.get_classes()

        # read image
        images = utility.read_images(
            images, target_size=TARGET_SIZE)

        with graph.get_instace().as_default():
            for image in images:
                image_match_objects = list()
                image_name = image['name']
                image_data = image['data']
                image_shape = image['shape']

                # predict objects
                objects_result = model.detect([image_data], verbose=0)[0]
                boxes = objects_result['rois'].tolist()
                classes_ids = objects_result['class_ids']
                scores = objects_result['scores']
                masks = objects_result['masks']
                N = len(objects_result['class_ids'])
                scores = scores.astype('float64').tolist()

                # get colors for objects
                colors = ColorPy.get_colors(image_data, masks)
                colors = np.asarray(colors, dtype=np.float32)

                for i in range(N):
                    score = round(scores[i], 2)
                    if score >= CONFIDENCE_THRESHOLD:
                        flag = True
                        y1, x1, y2, x2 = boxes[i]
                        box = utility.correct_detect_box(
                            [x1, y1, x2, y2], image_shape, TARGET_SIZE)
                        image_match_objects.append(DetectedBox(classes[classes_ids[i]],
                                                               box, score, base64.b64encode(colors[i]).decode('utf-8')).__dict__)

                match_result.append(DetectionResult(image_name, len(
                    image_match_objects), image_match_objects).__dict__)
        return result.success(match_result,
                              'no objects found!!!' if not flag else 'process done successfully!!!')
    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def recognition(json_data):
    try:
        MATCH_THRESHOLD = 0.5
        TARGET_SIZE = 11
        known_colors = utility.decode_embeddings(
            json_data['Known']['Embeddings'], TARGET_SIZE)
        candidates = json_data['Candidates']
        score_list = list()
        for candidate in candidates:
            candidate_colors = utility.decode_embeddings(
                candidate['Embeddings'], TARGET_SIZE)
            score = match(known_colors, candidate_colors)
            if score <= MATCH_THRESHOLD:
                score_list.append(MatchScore(
                    candidate['ItemId'], score).__dict__)

        score_list.sort(key=lambda x: x['score'])
        return result.success(MatchResult(score_list, known_colors).__dict__, 'no matches found!' if len(score_list) == 0 else 'Matches!')

    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def match(known_colors, candidate_colors):
    # calculate distance between colors
    score = cosine(known_colors, candidate_colors)
    return score
