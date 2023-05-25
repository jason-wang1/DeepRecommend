import tensorflow as tf

from layers.deepfm_tower import TowerDeepFM
from models.base import Base


class TwoTowerDeepFM(Base):
    def __init__(self, config, **kwargs):
        super(TwoTowerDeepFM, self).__init__(config, **kwargs)

    def build(self, input_shape):
        assert len(self.config["data_config"]["feature_groups"]) == 2
        user_input_dim = self.config["model_config"]["max_global_index"]["user"] + 1
        item_input_dim = self.config["model_config"]["max_global_index"]["item"] + 1
        emb_dim = self.config["model_config"]["embedding_dim"]
        dnn_shape = self.config["model_config"]["deep_hidden_units"]
        self.user_tower = TowerDeepFM("user", user_input_dim, emb_dim, dnn_shape, self.reg, name="user_tower")
        self.item_tower = TowerDeepFM("item", item_input_dim, emb_dim, dnn_shape, self.reg, name="item_tower")

    def call(self, inputs, training=None, mask=None):
        user_represent = self.user_tower(inputs["user"])  # (batch_size, represent_size)
        item_represent = self.item_tower(inputs["item"])  # (batch_size, represent_size)
        pred = tf.reduce_sum(user_represent * item_represent, axis=1, keepdims=True)
        pred = tf.sigmoid(pred)
        return pred
