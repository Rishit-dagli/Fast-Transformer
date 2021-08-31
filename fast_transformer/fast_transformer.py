import tensorflow as tf


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.mult = mult

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, input_dim=dim),
                tf.keras.layers.Activation(tf.nn.gelu),
                tf.keras.layers.Dense(dim, input_dim=dim * mult),
            ]
        )

    def call(self, inputs, **kwargs):
        return self.net(inputs)

class FastTransformer(tf.keras.Model):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        absolute_pos_emb = False,
        mask = None):
        super(FastTransformer, self).__init__()

    def call(self, x, **kwargs):