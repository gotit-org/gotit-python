
class FaceDetected:
    def __init__(self, imageName, count, faces):
        self.imageName = imageName
        self.count = count
        self.faces = faces


class FaceVerificationResult:
    def __init__(self, matching_score: list, embeddenges: str):
        self.matching_score = matching_score
        self.embeddenges = embeddenges


class ObjectDetected:
    def __init__(self, label, box, confidence, colors):
        self.label = label
        self.box = box
        self.confidence = confidence
        self.colors = colors


class ObjectDetectionResult:
    def __init__(self, count: int, objects: list):
        self.count = count
        self.objects = objects


class MatchScore:
    def __init__(self, id, score):
        self.id = id
        self.score = score
        