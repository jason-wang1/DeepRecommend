import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.regularizers import l1, l2

from layers.deepfm_tower import TowerDeepFM
from pipeline import TrainPipeline


def rebuild_deepfm_tower(group):
    input_dim = config["model_config"]["max_global_index"][group] + 1
    emb_dim = config["model_config"]["embedding_dim"]
    dnn_shape = config["model_config"]["deep_hidden_units"]
    if "l2_reg" in config["model_config"]:
        reg = l2(config["model_config"]["l2_reg"])
    elif "l1_reg" in config["model_config"]:
        reg = l1(config["model_config"]["l1_reg"])
    else:
        reg = None

    tower = TowerDeepFM(group, input_dim, emb_dim, dnn_shape, reg, name=f"{group}_tower")

    inputs = {"input_index": Input(shape=(pipeline.all_pad_num[group],), dtype=tf.int32),
              "input_value": Input(shape=(pipeline.all_pad_num[group],), dtype=tf.float32)}
    print(inputs)
    represent = tower(inputs)

    model = Model(inputs=inputs, outputs=represent)

    ori_model_var_dict = {}
    for var in ori_model.trainable_variables:
        ori_model_var_dict[var.name] = var

    for var in model.trainable_variables:
        org_var_name = 'two_tower_deep_fm/' + var.name
        var.assign(ori_model_var_dict[org_var_name])
    model.save(f"../output/{date_time}/{group}_model")

    return model


def check_result():
    dataset = pipeline.read_data("valid")
    for sample in dataset:
        features = sample[0]
        user_represent = user_model(features["user"])
        item_represent = item_model(features["item"])
        pred = tf.reduce_sum(user_represent * item_represent, axis=1, keepdims=True)
        pred = tf.sigmoid(pred)
        ori_pred = ori_model(features)
        print(pred)
        print(ori_pred)
        tf.assert_equal(pred, ori_pred)
        break


if __name__ == '__main__':
    date_time = "2023-05-24 164754"
    model_path = f"../output/{date_time}\\model"
    ori_model = keras.models.load_model(model_path)
    var_names = [var.name for var in ori_model.trainable_variables]
    print("ori_model:", var_names)

    feat_config_path = f"../output/{date_time}/input_fields.json"
    model_config_path = f"../output/{date_time}/model_config.json"
    pipeline = TrainPipeline(feat_config_path, model_config_path, run_eagerly=False)
    pipeline.config["data_config"]['valid_limit'] = 1024
    pipeline.config["data_config"]["batch_size"] = 32
    config = pipeline.config

    user_model = rebuild_deepfm_tower("user")
    item_model = rebuild_deepfm_tower("item")
    check_result()
