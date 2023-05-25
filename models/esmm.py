import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Flatten

from layers.dnn import DNN
from models.base import Base


class ESMM(Base):
    def __init__(self, config, **kwargs):
        super(ESMM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        assert len(self.config["data_config"]["feature_groups"]) == 1
        input_dim = self.config["model_config"]["max_global_index"]["all"] + 1
        self.emb = Embedding(input_dim, self.emb_dim, embeddings_regularizer=self.reg, name=f'emb')
        self.flatten = Flatten()
        dnn_shape = self.config["model_config"]["deep_hidden_units"]
        self.ctr_tower = DNN(dnn_shape=dnn_shape, reg=self.reg, name="ctr_tower")
        self.cvr_tower = DNN(dnn_shape=dnn_shape, reg=self.reg, name="cvr_tower")

    def call(self, inputs, training=None, mask=None):
        dnn_input = self.emb(inputs["input_index"])  # (batch_size, feat_size, emb_dim)
        dnn_input = dnn_input * tf.expand_dims(inputs["input_value"], axis=2)
        dnn_input = self.flatten(dnn_input)  # (batch_size, feat_size * emb_dim)
        ctr_pred = self.ctr_tower(dnn_input)  # (batch_size, 1)
        ctr_pred = tf.squeeze(tf.sigmoid(ctr_pred), axis=1)  # (batch_size,)
        cvr_pred = self.cvr_tower(dnn_input)  # (batch_size, 1)
        cvr_pred = tf.squeeze(tf.sigmoid(cvr_pred), axis=1)  # (batch_size,)
        ctcvr_pred = ctr_pred * cvr_pred
        return ctr_pred, ctcvr_pred
