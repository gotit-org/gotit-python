from modules.singleton import mrcnn, graph
from modules import utility
from models import result
from models.response_models import ObjectDetected, ObjectDetectionResult
from setuptools import sandbox
import os
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

        # load model & classes
        model = mrcnn.get_model()
        classes = mrcnn.get_classes()

        # read image
        img = utility.read_images(
            image, target_size=(448, 448), require_count=1)
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

            for i in range(N):
                final_data.append(ObjectDetected(classes[classes_ids[i]],
                                                 boxes[i], round(scores[i], 2), colors[i]).__dict__)

        return result.success(ObjectDetectionResult(len(final_data), final_data).__dict__,
                              'no objects found!!!' if len(final_data) == 0 else 'process done successfully!!!')
    except Exception as e:
        return result.failed(data=None, message=str(e), status_code=e.code if hasattr(e, 'code') else 500)


def match():
    pass
