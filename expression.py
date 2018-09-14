import tensorflow as tf

inputs = tf.placeholder(shape=[None, 2304], dtype=tf.float32)


def expression_nn(inputs):
    x = tf.reshape(inputs, shape=[-1, 48, 48, 1])
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        name='expression_conv1')
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same',
        name='expression_pool1')
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        name='expression_conv2')
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same',
        name='expression_pool2')
    
    norm2 = tf.nn.lrn(
        input=pool2,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75)
    
    dense1 = tf.layers.dense(
        inputs=tf.reshape(norm2, -1, 12 * 12 * 64),
        units=384,
        activation=tf.nn.relu,
        name='expression_dense1'
    )
    
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=192,
        name='expression_dense2'
    
    )
    
    logits = tf.layers.dense(
        inputs=dense2,
        units=7,
        name='expression_logits'
    )
    return logits


for v in tf.trainable_variables():
    print(v)

tensors = load_tensors(LANDMARK_TENSORS_PATH)
print(tensors.keys())

sess = tf.Session()
for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(v)
    v.load(tensors[v.name], sess)
