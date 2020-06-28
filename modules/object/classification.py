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


def detect(image):
    try:
        # init result list
        final_data = list()
        match_result = list()

        # load model & classes
        model = mrcnn.get_model()
        classes = mrcnn.get_classes()

        # read image
        img = utility.read_images(
            image, target_size=(448, 448), require_count=1)
        imageName = img[0]['name']
        img = img[0]['image']

        # detect objects
        with graph.get_instace().as_default():
            # predict objects
            objects_result = model.detect([img], verbose=0)[0]
            boxes = objects_result['rois'].tolist()
            classes_ids = objects_result['class_ids']
            scores = objects_result['scores']
            masks = objects_result['masks']
            N = len(objects_result['class_ids'])
            scores = scores.astype('float64').tolist()

            # get colors for objects
            colors = ColorPy.get_colors(img, masks)
            colors = np.asarray(colors, dtype=np.float32)

            for i in range(N):
                final_data.append(DetectedBox(classes[classes_ids[i]],
                                                 boxes[i], round(scores[i], 2), base64.b64encode(colors[i]).decode('utf-8')).__dict__)
        match_result.append(DetectionResult(imageName, len(final_data), final_data).__dict__)
        return result.success(match_result,
                              'no objects found!!!' if len(final_data) == 0 else 'process done successfully!!!')
    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def recognition(json_data):
    try:
        MATCH_THRESHOLD = 0.5
        known_colors = decode(json_data['known'])['embedding']
        candidates = json_data['candidates']
        score_list = list()
        for candidate in candidates:
            candidate_colors = decode(candidate['embedding'])
            score = match(known_colors, candidate_colors)
            if score <= MATCH_THRESHOLD:
                score_list.append(MatchScore(candidate['id'], score).__dict__)

        score_list.sort(key=lambda x: x['score'])
        return result.success(MatchResult(score_list, None).__dict__, 'no matches found!' if len(score_list) == 0 else 'Matches!!!')

    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def match(known_colors, candidate_colors):
    # calculate distance between colors
    score = cosine(known_colors, candidate_colors)
    return score


def decode(encode_data):
    return np.fromstring(base64.b64decode(encode_data), np.float32)
