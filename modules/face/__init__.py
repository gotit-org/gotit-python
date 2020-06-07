from .verification import extract_face, recognition
from modules.singleton import vgg, mtcnn, graph


# create a mtcnn model
print("start load mtcnn model")
mtcnn.creat_instance()

# create a vggface model
print("start load vgg model")
vgg.creat_instance()

graph.creat_instance()