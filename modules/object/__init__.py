from .classification import detect, match
from modules import utility
from modules.singleton import mrcnn, graph
import os
import glob
from distutils.dir_util import remove_tree


# Download weights if not exists
DOWNLOADS_DIR = os.path.dirname(os.path.abspath(__file__))
url = 'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5'
name = 'mask_rcnn_coco.h5'
# Combine the name and the downloads directory to get the local filename
file_path = os.path.join(DOWNLOADS_DIR, name)
# Download the file if it does not exist
if not os.path.isfile(file_path):
    utility.download_file(url, name, file_path)

# create a mtcnn model
print("start load mrcnn model")
mrcnn.creat_instance()

graph.creat_instance()

# remove temp directory 
fileList = glob.glob(DOWNLOADS_DIR + '/temp*/', recursive=True)
for filePath in fileList:
    try:
        print(filePath)
        if os.path.isdir(filePath):
            # os.remove(filePath)
            remove_tree(filePath)
    except OSError as e:
        print("Error while deleting file" + str(e))