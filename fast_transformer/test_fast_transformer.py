import numpy as np
import tensorflow as tf

from fast_transformer.fast_transformer import FastTransformer


class FastTransformerTest(tf.test.TestCase):
    def setUp(self):
        super(FastTransformerTest, self).setUp()

        mask = tf.ones([1, 4096], dtype=tf.bool)
        self.model = FastTransformer(
            num_tokens=20000,
            dim=512,
            depth=2,
            max_seq_len=4096,
            absolute_pos_emb=True,
            mask=mask,
        )

    def test_shape_and_rank(self):
        inputs = tf.experimental.numpy.random.randint(0, 20000, (1, 4096))
        outputs = self.model(inputs)

        self.assertEqual(tf.rank(outputs), 3)
        self.assertShapeEqual(np.zeros((1, 4096, 20000)), outputs)


if __name__ == "__main__":
    tf.test.main()
