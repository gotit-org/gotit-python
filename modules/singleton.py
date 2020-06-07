from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import tensorflow as tf
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import os


class mtcnn:
    _model = None

    @staticmethod
    def creat_instance():
        if mtcnn._model == None:
            mtcnn._model = MTCNN()

    @staticmethod
    def get_instace():
        if mtcnn._model == None:
            mtcnn.creat_instance()

        return mtcnn._model


class vgg:
    _model = None

    @staticmethod
    def creat_instance():
        if vgg._model == None:
            vgg._model = VGGFace(model='resnet50', include_top=False,
                                   input_shape=(224, 224, 3), pooling='avg')

    @staticmethod
    def get_instace():
        if vgg._model == None:
            vgg.creat_instance()

        return vgg._model


class graph:
    _defualt = None

    @staticmethod
    def creat_instance():
        if graph._defualt == None:
            graph._defualt = tf.get_default_graph()

    @staticmethod
    def get_instace():
        if graph._defualt == None:
            graph.creat_instance()

        return graph._defualt


class MRCNNConfig(Config):
    NAME = "temp"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448


class mrcnn:
    _model = None

    @staticmethod
    def creat_instance():
        if mrcnn._model == None:
            MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + '/object'
            mrcnn._model = MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=MRCNNConfig())
            mrcnn._model.load_weights(MODEL_DIR + '/mask_rcnn_coco.h5', by_name=True)

    @staticmethod
    def get_model():
        if mrcnn._model == None:
            mrcnn.creat_instance()

        return mrcnn._model

    @staticmethod
    def get_classes():
        return ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
