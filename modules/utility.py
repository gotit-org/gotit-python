from tqdm import tqdm
import requests
import cv2
import numpy as np
import json
import base64


class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code

    def __str__(self):
        return '{} - {}'.format(self.code, self.message)


def allowed_file(filename, ALLOWED_EXTENSIONS) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_body(body: dict, data_key: str):
    data = json.loads(body[data_key])
    return data


def read_images(files, target_size=None, require_count=None, limit: int=5) -> list:
    try:
        if limit < len(files):
            raise CustomError(
                message="You have exceeded the images limit. you can only upload {} image".format(
                    limit),
                code=413
            )

        if require_count != None and require_count != len(files):
            raise CustomError(
                message="You have sent an inappropriate number of images. The number of images must be {}".format(
                    require_count),
                code=400
            )

        images = list()
        for file in files.keys():
            fileData = files[file]
            images.append(read_image(files[file], target_size))

        return images

    except Exception as e:
        raise e


def read_b64image(b64image):
    image = base64.b64decode(b64image)
    image = cv2.imdecode(np.fromstring(
        image, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_image(fileData, target_size=None):
    result = None
    if fileData.filename == '':
        raise CustomError(message="file has no name", code=400)

    if allowed_file(fileData.filename, {'png', 'jpg', 'jpeg', 'gif'}):
        image = cv2.imdecode(np.fromstring(
            fileData.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # resize if there is a target size
        if target_size != None:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)


        result = {
            'name': fileData.filename,
            'image': image
        }

    else:
        raise CustomError(message="invalid file extension", code=422)

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
