import tensorflow as tf

class AbstractModel(object):
    """Abstract class for models"""
    def __init__(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = 0.001
        self.optimizer = tf.train.AdamOptimizer(self.lr)

    def build_net(self, X):
        return NotImplementedError

    def filter_vars(self, variables):
        return None
