import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

a = tf.constant(
    [
        [
            [[1, 1], [2, 3]], [[1, 1], [2, 3]], [[1, 1], [2, 3]], [[1, 1], [2, 3]]
        
        ],
        [
            [[1, 1], [2, 3]], [[1, 1], [2, 3]], [[1, 1], [2, 3]], [[1, 1], [2, 3]]
        
        ]
    ], dtype=tf.float32)
print(a)

b = tf.constant(
    [
        [
            [3, 4, 5, 6], [3, 8, 5, 6], [3, 1, 5, 6], [3, 9, 5, 6]
        ],
        [
            [3, 4, 5, 6], [3, 8, 5, 6], [3, 1, 5, 6], [3, 9, 5, 6]
        ]
    ], dtype=tf.float32)
print('b', b)
print('bbb', b[:, :, 1])
# c = b[:, :, 1]
# print(c)
# print(tf.tile(, [2, 1]))

c = tf.expand_dims(b[:, :, 1], 1)
d = tf.expand_dims(b[:, :, 2], 1)
print('c', c)
print('d', d)
e = tf.concat([c, d], axis=1)

print('e', e)
f = tf.transpose(e, [0, 2, 1])
print('f', f)

g = tf.expand_dims(f, 2)
print('g', g)

h = tf.tile(g, [1, 1, 2, 1])
print('h', h)
# print(a[:, :, :, 1])
# print(c)
# b[:, :, 1] += 2
# print(b)

result = a + h
print(result)
