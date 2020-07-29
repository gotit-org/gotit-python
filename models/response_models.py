
class DetectedBox:
    def __init__(self, label: str, box, confidence: float, embeddenges: str):
        self.label = label
        self.box = box
        self.confidence = confidence
        self.embeddenges = embeddenges


class DetectionResult:
    def __init__(self, imageName: str, count: int, detectd: list):
        self.imageName = imageName
        self.count = count
        self.detectd = detectd


class ObjectDetectionResult:
    def __init__(self, count: int, objects: list):
        self.count = count
        self.objects = objects


class MatchScore:
    def __init__(self, itemId, score):
        self.itemId = itemId
        self.score = score


class MatchResult:
    def __init__(self, scores: list, embeddings: str):
        self.scores = scores
        self.embeddings = embeddings
