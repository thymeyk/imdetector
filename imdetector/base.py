import inspect
import cv2 as cv
from enum import IntEnum


class Color(IntEnum):
    B = 0
    G = 1
    R = 2
    Y = 3
    Cr = 4
    Cb = 5


class DrawFlags(IntEnum):
    SHOW_RESULT = 0
    SHOW_FULL_RESULT = 1
    RETURN_RESULT = 2


class BaseDetector:
    """
    Base class for all detectors
    """

    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def get_params(self):
        """
        Get parameters for this estimator.
        """

        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            # if deep and hasattr(value, 'get_params'):
            #     deep_items = value.get_params().items()
            #     out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def __repr__(self):
        params = self.get_params()
        repr_ = ''
        for name, val in params.items():
            repr_ += '{}={}, '.format(name, val)
        repr_ = repr_.rsplit(',', 1)[0]
        return '{}({})'.format(self.__class__.__name__, repr_)

    def save_image(self, file_name):
        if hasattr(self, 'image_'):
            cv.imwrite(file_name, self.image_)
        else:
            print('Error: There is no result image.')
        return self
