from collections import OrderedDict
import torch.nn as nn


class NamedSequential(nn.Sequential):
    """A version of nn.Sequential that uses named arguments without needing
    to explicitly use an OrderedDict"""

    def __init__(self, **kwargs):
        """Note: the order of kwargs is preserved as of PEP 468:
        https://www.python.org/dev/peps/pep-0468/"""
        super().__init__(OrderedDict(kwargs))
