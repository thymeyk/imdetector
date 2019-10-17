import inspect
from enum import Enum


class DrawFlags(Enum):
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
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
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
