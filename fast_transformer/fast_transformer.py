import tensorflow as tf


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x)
