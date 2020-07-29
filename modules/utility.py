from tqdm import tqdm
import requests
import cv2
import numpy as np
import json
import base64
import urllib.request


class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code

    def __str__(self):
        return '{} - {}'.format(self.code, self.message)


def allowed_file(filename, ALLOWED_EXTENSIONS) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_images(images, target_size=None, require_count=None, limit: int=5) -> list:
    try:
        if require_count != None and require_count != len(images):
            raise CustomError(
                message="You have sent an inappropriate number of images. The number of images must be {}".format(
                    require_count),
                code=400
            )

        result = list()
        for image in images:
            result.append(read_image(image, target_size))

        return result

    except Exception as e:
        raise e


def read_image(fileData, target_size=None):
    result = None
    filename = fileData.rsplit('/', 1)[1].lower()
    if filename == '':
        raise CustomError(message="file has no name", code=400)
    
    if not(allowed_file(filename, {'png', 'jpg', 'jpeg'})):
        raise CustomError(message="invalid file extension", code=422)
    
    local_filename, headers = urllib.request.urlretrieve(fileData)
    image = cv2.imread(local_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orignial_height = image.shape[0]
    orignial_width = image.shape[1]
    
    # resize if there is a target size
    if target_size != None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    result = {
        'name': filename,
        'data': image,
        'shape': (orignial_height, orignial_width)
    }

    return result


def download_file(url, filename, path):
    print("start downloading {}".format(filename))
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")


def decode_embeddings(encode_embeddings, TARGET_SIZE):
    decoded = fromstring(base64.b64decode(encode_embeddings), float32)
    n = int(decoded.shape[0] / TARGET_SIZE)
    return reshape(decoded, (n, TARGET_SIZE))


def correct_detect_box(box, original_shape, network_shape):
    image_h, image_w = original_shape
    net_h, net_w = network_shape
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
    
    box[0] = int((box[0] - x_offset) / x_scale * image_w)
    box[1] = int((box[1] - y_offset) / y_scale * image_h)
    box[2] = int((box[2] - x_offset) / x_scale * image_w)
    box[3] = int((box[3] - y_offset) / y_scale * image_h)

    # return [x1, x2, width, height]
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
