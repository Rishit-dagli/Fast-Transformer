import tensorflow as tf
from fast_transformer import FastTransformer

mask = tf.ones([1, 4096], dtype=tf.bool)
model = FastTransformer(
    num_tokens=20000,
    dim=512,
    depth=2,
    max_seq_len=4096,
    absolute_pos_emb=True,  # Absolute positional embeddings
    mask=mask,
)
x = tf.experimental.numpy.random.randint(0, 20000, (1, 4096))

logits = model(x)  # (1, 4096, 20000)
print("Should be (1, 4096, 20000):", logits.shape)
