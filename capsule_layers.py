# import tensorflow as tf

# def squash(vectors, axis=-1):
#     s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
#     scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
#     return scale * vectors

# class CapsuleLayer(tf.keras.layers.Layer):
#     def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
#         super(CapsuleLayer, self).__init__(**kwargs)
#         self.num_capsule = num_capsule
#         self.dim_capsule = dim_capsule
#         self.routings = routings

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='capsule_kernel',
#                                       shape=(input_shape[-1], self.num_capsule * self.dim_capsule),
#                                       initializer='glorot_uniform',
#                                       trainable=True)

#     def call(self, inputs):
#         if len(inputs.shape) != 4:
#             raise ValueError(f"Expected inputs to CapsuleLayer to be 4D, received shape {inputs.shape}")

#         inputs_expand = tf.expand_dims(inputs, 1)
#         print("After expand_dims:", inputs_expand.shape)

#         inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
#         print("After tiling:", inputs_tiled.shape)

#         inputs_transformed = tf.reshape(tf.matmul(inputs_tiled, self.kernel), [-1, self.num_capsule, self.dim_capsule])
#         print("After transformation:", inputs_transformed.shape)

#         b = tf.zeros(shape=[tf.shape(inputs_transformed)[0], self.num_capsule])
#         for i in range(self.routings):
#             c = tf.nn.softmax(b, axis=1)
#             s = tf.reduce_sum(c[:, :, None] * inputs_transformed, axis=1)
#             v = squash(s)
#             print(f"Output v after routing {i+1}:", v.shape)
#             if i < self.routings - 1:
#                 b += tf.matmul(inputs_transformed, v[:, None, :], transpose_b=True)

#         return v

#     def compute_output_shape(self, input_shape):
#         return (None, self.num_capsule, self.dim_capsule)


# class Mask(tf.keras.layers.Layer):
#     def call(self, inputs, **kwargs):
#         input, mask = inputs
#         return tf.keras.backend.batch_dot(input, mask, [2, 1])

#     def compute_output_shape(self, input_shape):
#         return tuple(input_shape[0])

# def margin_loss(y_true, y_pred):
#     L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
#         0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
#     return tf.reduce_mean(tf.reduce_sum(L, axis=1))
