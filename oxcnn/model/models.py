class AbstractModel(object):
    """Abstract class for models"""

    def build_net(self, X):
        return NotImplementedError

    def filter_vars(self, variables):
        return None
