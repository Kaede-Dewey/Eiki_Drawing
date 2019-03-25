"""
this file proposes the utils to
analyze picture we write into 14 vectors.
"""
import cv2


class Image2Vec(object):

    def __init__(self, img_path):
        self.img_vector = self._get_img_vector(img_path)

    def _get_img_vector(self, path):
        return [cv2.imread(path, 0)]
