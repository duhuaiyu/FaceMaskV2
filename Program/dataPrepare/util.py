import cv2
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional

import random

def get_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class Handler(ABC):
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    """

    @abstractmethod
    def set_next(self, handler) :
        pass

    @abstractmethod
    def handle(self, request) -> Optional[str]:
        pass


class AbstractHandler(Handler):
    """
    The default chaining behavior can be implemented inside a base handler
    class.
    """

    _next_handler: Handler = None

    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        # Returning a handler from here will let us link handlers in a
        # convenient way like this:
        # monkey.set_next(squirrel).set_next(dog)
        return handler


    def handle(self, image: Any) -> str:
        res = self.process(image)
        if self._next_handler:
            return self._next_handler.handle(res)
        return res

    @abstractmethod
    def process(self,image):
        pass


"""
All Concrete Handlers either handle a request or pass it to the next handler in
the chain.
"""


class FlipHandler(AbstractHandler):
    def process(self, image):
        if (bool(random.getrandbits(1))):
            return cv2.flip(image,1)
        else:
            return image


class RotationHandler(AbstractHandler):
    def process(self, image: Any) -> str:
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        if (bool(random.getrandbits(1))):
            degree = random.randint(-15, 15)
            M = cv2.getRotationMatrix2D(center, degree, scale=1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            return rotated
        else:
            return image

size = (32,32)
class ChopHandler(AbstractHandler):
    def process(self, image: Any) -> str:
        # return image
        (h, w) = image.shape[:2]
        c_h = int(h * 0.9)
        c_w = int(w * 0.9)
        s_h = random.randint(0, h - c_h)
        s_w = random.randint(0, w - c_w)
        return image[s_h:s_h+c_h,s_w:s_w+c_w]

class ResizeHandler(AbstractHandler):
    def process(self, image: Any) -> str:
        h, w = image.shape[:2]
        if random.random() < 0.2 and h > 10 and w > 10:

            r_h = int(h * 0.5)
            r_w = int(w * 0.5)
            return cv2.resize(cv2.resize(image,(r_h,r_w)),size)
        else:
            return cv2.resize(image,size)