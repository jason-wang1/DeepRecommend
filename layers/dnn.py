from tensorflow.python.keras.layers import Layer, Dense


class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. For instance,
        for a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance,
        for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
    """
    def __init__(self, dnn_shape, reg, activation='relu', final_activation=None, **kwargs):
        self.dnn_shape = dnn_shape
        self.reg = reg
        self.activation = activation
        self.final_activation = final_activation
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.deep_dense = []
        for i in range(len(self.dnn_shape) - 1):
            self.deep_dense.append(Dense(self.dnn_shape[i], activation=self.activation, kernel_regularizer=self.reg,
                                         bias_regularizer=self.reg, name=f'deep_dense_{i + 1}'))
        self.deep_dense.append(Dense(self.dnn_shape[-1], activation=self.final_activation, kernel_regularizer=self.reg,
                                     bias_regularizer=self.reg, name=f'deep_dense_{len(self.dnn_shape)}'))
        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        tensor = inputs
        for i in range(len(self.deep_dense)):
            tensor = self.deep_dense[i](tensor)
        return tensor
