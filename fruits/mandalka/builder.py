from inspect import signature

from .node import arguments, lazy

class NodeBuilder:

    cls = None

    def __init__(self, **params):
        pass

    @lazy
    def __call__(self, **more_params):
        params = arguments(self).copy()
        params.update(more_params)
        if len(more_params) == 0:
            # Force building
            return self.cls(**params)
        elif set(params.keys()) == set(self.get_keywords(self.cls)):
            # All params passed, build
            return self.cls(**params)
        elif set(params.keys()).issubset(set(self.get_keywords(self.cls))):
            # Some params still missing, return builder
            return self.__class__(**params)
        else:
            bad_args = set(params.keys()) - set(self.get_keywords(self.cls))
            raise ValueError("Invalid parameters: " + str(bad_args))

    @lazy
    def get_keywords(self, cls):
        if "mandalka.node" in str(cls):
            assert len(cls.__bases__) == 1
            inspectable_cls = cls.__bases__[0]
        else:
            inspectable_cls = cls
        keywords = []
        for kw, param in signature(inspectable_cls).parameters.items():
            assert param.kind in [
                param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY]
            keywords.append(kw)
        return keywords
