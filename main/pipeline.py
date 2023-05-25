import sys
import time
import math
from collections import OrderedDict
import json
import tensorflow as tf
from tensorflow.python.keras import optimizers
from models.esmm import ESMM
from models.twotower_deepfm import TwoTowerDeepFM


class TrainPipeline:
    def __init__(self, feat_config_path, model_config_path, run_eagerly=None):
        self.all_pad_num = OrderedDict()  # {feat_group_name: all_pad_num}
        self.feat_pad_num_dict = OrderedDict()  # {feat_group_name: {feat_field: (feat_start_index, pad_num)}}
        self.index_dict = OrderedDict()  # {feat_group_name: {feat_field: {feat_index: global_index}}}
        self.date_time = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        with open(feat_config_path) as f:
            self.feat_config_str = f.read()
            self.feat_config = json.loads(self.feat_config_str, object_pairs_hook=OrderedDict)
        with open(model_config_path) as f:
            self.model_config_str = f.read()
            model_config = json.loads(self.model_config_str, object_pairs_hook=OrderedDict)
        self.config = model_config
        self.feature_name_list = [feature_name for feature_name in self.config["data_config"]["feature_groups"].keys()]
        if len(self.feature_name_list) == 1:
            assert "all" in self.feature_name_list
        elif len(self.feature_name_list) == 2:
            assert "user" in self.feature_name_list
            assert "item" in self.feature_name_list
        else:
            raise ValueError(f"unexpected feature_groups length: {len(self.feature_name_list)}")
        self.config["model_config"]["max_global_index"] = OrderedDict()  # {feat_group_name: emb_input_dim}
        self.get_global_feature()
        self.run_eagerly = run_eagerly
        self.valid_data = None
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def get_global_feature(self):
        same_emb = self.config["data_config"].get("same_emb", {})
        for feat_group_name, feat_list in self.config["data_config"]["feature_groups"].items():
            global_index = 1
            self.all_pad_num[feat_group_name] = 0
            self.feat_pad_num_dict[feat_group_name] = OrderedDict()
            self.index_dict[feat_group_name] = OrderedDict()
            for feat in feat_list:
                feat_field = feat[0]
                pad_num = feat[1]
                self.feat_pad_num_dict[feat_group_name][feat_field] = (self.all_pad_num[feat_group_name], pad_num)
                self.all_pad_num[feat_group_name] += pad_num
                feat_attr = self.feat_config[str(feat_field)]
                if "boundaries" in feat_attr:
                    max_num = len(feat_attr["boundaries"])
                elif "hash_max_num" in feat_attr:
                    max_num = feat_attr["hash_max_num"]
                elif "max_num" in feat_attr:
                    max_num = feat_attr["max_num"]
                else:
                    raise ValueError(f"missing dict key")
                if str(feat_field) in same_emb:
                    self.index_dict[feat_group_name][feat_field] = self.index_dict[feat_group_name][int(same_emb[str(feat_field)])]
                else:
                    self.index_dict[feat_group_name][feat_field] = OrderedDict()
                    for feat_index in range(max_num+1):
                        self.index_dict[feat_group_name][feat_field][feat_index] = global_index
                        global_index += 1
            self.config["model_config"]["max_global_index"][feat_group_name] = global_index

    def encode_one_feature(self, record, feature_name):
        def append_one_feat(res, feat_value, feat_attr):
            def get_feat_index(value, feat_attr):
                if "max_num" in feat_attr:
                    return value
                elif "hash_max_num" in feat_attr:
                    return value & feat_attr["hash_max_num"]
                elif "boundaries" in feat_attr:
                    for i, boundary in enumerate(feat_attr["boundaries"]):
                        if value < boundary:
                            return i
                    return len(feat_attr["boundaries"])
                else:
                    raise ValueError(f"missing key on feat_attr")
            feat_index = get_feat_index(feat_value, feat_attr)
            global_index = self.index_dict[feature_name][feat_field].get(feat_index)
            if global_index:
                res["input_index"].append(global_index)
                res["input_value"].append(1.0)
            else:
                res["input_index"].append(0)
                res["input_value"].append(0.0)

        # 编码特征
        res = {"input_index": [], "input_value": []}
        for feat_field, (feat_start_index, pad_num) in self.feat_pad_num_dict[feature_name].items():
            feat_attr = self.feat_config[str(feat_field)]
            aliyun_feat_value = record[f"feat_{feat_field}"]
            if aliyun_feat_value:
                if feat_attr["field_type"] == "STRING":
                    i = 0
                    for feat_value in aliyun_feat_value.split(chr(1)):
                        append_one_feat(res, int(feat_value), feat_attr)
                        i += 1
                        if i == pad_num:
                            break
                    while i < pad_num:
                        res["input_index"].append(0)
                        res["input_value"].append(0.0)
                        i += 1
                else:
                    assert pad_num == 1
                    append_one_feat(res, aliyun_feat_value, feat_attr)
            else:
                if feat_attr["field_type"] == "STRING":
                    res["input_index"].extend([0 for _ in range(pad_num)])
                    res["input_value"].extend([0.0 for _ in range(pad_num)])
                else:
                    assert pad_num == 1
                    if feat_attr["field_type"] == "BIGINT":
                        aliyun_feat_value = 0
                    elif feat_attr["field_type"] == "DOUBLE":
                        aliyun_feat_value = 0.0
                    append_one_feat(res, aliyun_feat_value, feat_attr)
        assert len(res["input_index"]) == self.all_pad_num[feature_name]
        assert len(res["input_value"]) == self.all_pad_num[feature_name]
        return res

    def read_aliyun_data(self, data_type):

        def read_max_compute():
            from config import access_id, secret_access_key, project, endpoint
            from odps import ODPS
            o = ODPS(access_id, secret_access_key, project, endpoint=endpoint)

            with o.execute_sql(sql).open_reader() as reader:
                for record in reader:
                    # 编码特征
                    if len(self.feature_name_list) == 1:
                        res_features = self.encode_one_feature(record, "all")
                    elif len(self.feature_name_list) == 2:
                        res_features = {"user": self.encode_one_feature(record, "user"),
                                        "item": self.encode_one_feature(record, "item")}
                    # 编码标签
                    if len(labels) == 1:
                        _label_name = list(labels.keys())[0]
                        res_labels = record[_label_name]
                    else:
                        res_labels = {}
                        for i, _label_name in enumerate(labels):
                            res_labels[f"output_{i + 1}"] = record[_label_name]
                    yield res_features, res_labels

        labels = self.config["data_config"]["label_list"]
        if len(self.feature_name_list) == 1:
            output_types_features = {"input_index": tf.int32, "input_value": tf.float32}
            output_shape_features = {"input_index": [self.all_pad_num["all"], ], "input_value": [self.all_pad_num["all"], ]}
        else:
            output_types_features = {"user": {"input_index": tf.int32, "input_value": tf.float32},
                                     "item": {"input_index": tf.int32, "input_value": tf.float32}}
            output_shape_features = {"user": {"input_index": [self.all_pad_num["user"], ], "input_value": [self.all_pad_num["user"], ]},
                                     "item": {"input_index": [self.all_pad_num["item"], ], "input_value": [self.all_pad_num["item"], ]}}
        if len(labels) == 1:
            output_shape_labels = []
            label_name = list(labels.keys())[0]
            if labels[label_name] == "BIGINT":
                output_types_labels = tf.int32
            elif labels[label_name] == "DOUBLE":
                output_types_labels = tf.float32
        else:
            output_shape_labels = {}
            output_types_labels = {}
            for i, label_name in enumerate(labels):
                output_shape_labels[f"output_{i + 1}"] = []
                if labels[label_name] == "BIGINT":
                    output_types_labels[f"output_{i + 1}"] = tf.int32
                elif labels[label_name] == "DOUBLE":
                    output_types_labels[f"output_{i + 1}"] = tf.float32
        if data_type == "train":
            sql = f"""
            SELECT  *
            FROM    {self.config["data_config"]['table_name']}
            WHERE   dt BETWEEN {self.config["data_config"]['train_start_dt']} AND {self.config["data_config"]['train_end_dt']}
            ORDER BY RAND()
            """
            dataset = tf.data.Dataset.from_generator(read_max_compute, output_types=(output_types_features, output_types_labels), output_shapes=(output_shape_features, output_shape_labels))
        elif data_type == "valid":
            if not self.valid_data:
                sql = f"""
                SELECT  *
                FROM    {self.config["data_config"]['table_name']}
                WHERE   dt BETWEEN {self.config["data_config"]['valid_start_dt']} AND {self.config["data_config"]['valid_end_dt']}
                LIMIT   {self.config["data_config"]['valid_limit']}
                """
                self.valid_data = [sample for sample in read_max_compute()]

            def valid_data_gen():
                for sample in self.valid_data:
                    yield sample
            dataset = tf.data.Dataset.from_generator(valid_data_gen, output_types=(output_types_features, output_types_labels), output_shapes=(output_shape_features, output_shape_labels))
        else:
            raise ValueError(f"unexpected data_type: {data_type}")

        return dataset

    def read_data(self, data_type):
        data_config = self.config["data_config"]
        if data_config["input_type"] == "MaxComputeInput":
            train_ds = self.read_aliyun_data(data_type)
        else:
            raise ValueError(f"unexpected input_type: {data_config['input_type']}")
        if data_type == "train":
            train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
            train_ds = train_ds.repeat(self.config["data_config"].get("repeat", 1))
        train_ds = train_ds.batch(batch_size=self.config["data_config"]["batch_size"], drop_remainder=True)
        # train_ds = train_ds.map(self.map_sample(), num_parallel_calls=4)
        train_ds = train_ds.prefetch(data_config["prefetch_size"])
        return train_ds

    def get_optimizer(self):
        optimizer_config = self.config["train_config"]["optimizer"]
        learning_rate = optimizer_config["learning_rate"]
        if isinstance(learning_rate, float):
            lr_schedule = learning_rate
        else:
            raise ValueError(f"unexpected config type: {learning_rate}")
        if optimizer_config["name"] == "adam":
            optimizer = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
        else:
            raise ValueError(f"unexpected optimizer name: {optimizer_config['name']}")
        return optimizer

    def get_callbacks(self):
        patience = self.config["train_config"].get("early_stopping_patience", math.ceil(self.config["train_config"]["epochs"]/10))
        log_dir = f"..\\output\\{self.date_time}\\tensorboard"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            # update_freq=5000,
            histogram_freq=1
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=patience,
            restore_best_weights=True
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath="..\output\checkpoint_model",
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        return [tensorboard_callback, early_stopping_callback, checkpoint_cb]

    def get_model(self):
        optimizer = self.get_optimizer()
        model_type = self.config["model_config"]["model_type"]
        if model_type == "ESMM":
            model = ESMM(self.config)
        elif model_type == "TwoTowerDeepFM":
            model = TwoTowerDeepFM(self.config)
        else:
            print(f"unexpected model_type: {model_type}")
            sys.exit(1)
        if model_type in ["ESMM"]:
            model.compile(optimizer=optimizer, loss={"output_1": self.config["train_config"]["loss_1"], "output_2": self.config["train_config"]["loss_2"]},
                          metrics={"output_1": self.config["train_config"]["metrics_1"], "output_2": self.config["train_config"]["metrics_2"]},
                          run_eagerly=self.run_eagerly)
        else:
            model.compile(optimizer=optimizer, loss=self.config["train_config"]["loss"],
                          metrics=self.config["train_config"]["metrics"], run_eagerly=self.run_eagerly)
        return model

    def train(self):
        train_ds = self.read_data(data_type="train")
        valid_ds = self.read_data(data_type="valid")
        model = self.get_model()
        model.fit(
            x=train_ds,
            epochs=self.config["train_config"]["epochs"],
            callbacks=self.get_callbacks(),
            steps_per_epoch=self.config["train_config"]["steps_per_epoch"],
            validation_data=valid_ds
        )
        return model


# @tf.function
def get_one_sample():
    # 预览训练样本
    ds = pipeline.read_data(data_type="train")
    for sample in ds:
        for feature_name, feature_tensor in sample[0].items():
            print(f"{feature_name}: {feature_tensor}")
        print(sample[1])
        print(sample)
        break


# @tf.function
def get_one_sample_ori():
    # 预览训练样本
    ds = pipeline.read_data(data_type="train")
    for sample in ds:
        print(sample)
        break


def encode_one():
    # 特征编码测试用例
    # feature = '{"feat_1": 177451038, "feat_2": 0, "feat_3": 0, "feat_4": 0, "feat_5": 0, "feat_9": 20, "feat_10": 0, "feat_11": 0, "feat_12": 0, "feat_13": 4889110, "feat_15": 1, "feat_17": 7, "feat_18": 484, "feat_19": 751, "feat_20": 28, "feat_21": 1, "feat_24": 1, "feat_25": 2639339, "feat_26": 0, "feat_27": 2, "feat_28": 1, "feat_29": 1, "feat_30": 1, "feat_31": 0, "feat_32": 1, "feat_33": 0, "feat_34": 0, "feat_35": 0, "feat_36": 0, "feat_37": 1, "feat_38": 12, "feat_39": 28, "feat_40": 0.009716599190283401, "feat_41": 0.012956964368347987, "feat_42": 1, "feat_43": 3, "feat_44": 0.0008097165991902834, "feat_45": 0.0013882461823229986, "feat_46": 7783603, "feat_48": 1, "feat_50": 7, "feat_51": 484, "feat_52": 751, "feat_53": 28, "feat_54": 1, "feat_57": 1, "feat_58": 2639339, "feat_59": 0, "feat_60": 2, "feat_61": 1, "feat_62": 1, "feat_63": 1, "feat_64": 0, "feat_65": 1, "feat_66": 0, "feat_67": 0, "feat_68": 0, "feat_69": 0, "feat_70": 1, "feat_71": 12, "feat_72": 28, "feat_73": 0.009716599190283401, "feat_74": 0.012956964368347987, "feat_75": 1, "feat_76": 3, "feat_77": 0.0008097165991902834, "feat_78": 0.0013882461823229986}'
    feature = '{"feat_1": 902891, "feat_2": 1, "feat_3": 130000, "feat_4": 130700, "feat_5": 4, "feat_9": 180, "feat_10": 1, "feat_11": 5478, "feat_12": 0, "feat_13": 1009417, "feat_15": 1, "feat_17": 2, "feat_18": 512, "feat_19": 940, "feat_20": 28, "feat_21": 1, "feat_24": 1, "feat_25": 590386, "feat_26": 0, "feat_27": 4, "feat_28": 1, "feat_29": 1, "feat_30": 1, "feat_31": 0, "feat_32": 1, "feat_33": 0, "feat_34": 0, "feat_35": 0, "feat_36": 0, "feat_37": 1, "feat_38": 4, "feat_39": 12, "feat_40": 0.006622516556291391, "feat_41": 0.008247422680412371, "feat_42": 1, "feat_43": 1, "feat_44": 0.0016556291390728477, "feat_45": 0.0006872852233676976, "feat_46": 783983, "feat_48": 1, "feat_50": 2, "feat_51": 512, "feat_52": 940, "feat_53": 28, "feat_54": 1, "feat_57": 1, "feat_58": 590386, "feat_59": 0, "feat_60": 4, "feat_61": 1, "feat_62": 1, "feat_63": 1, "feat_64": 0, "feat_65": 1, "feat_66": 0, "feat_67": 0, "feat_68": 0, "feat_69": 0, "feat_70": 1, "feat_71": 4, "feat_72": 12, "feat_73": 0.006622516556291391, "feat_74": 0.008247422680412371, "feat_75": 1, "feat_76": 1, "feat_77": 0.0016556291390728477, "feat_78": 0.0006872852233676976}'
    feature = json.loads(feature)
    res = pipeline.encode_one_feature(feature)
    print(res)


if __name__ == '__main__':
    pipeline = TrainPipeline("../config/data_aliyun_down_rec/input_fields.json",
                             "../config/data_aliyun_down_rec/model_twotower_deepfm.json", run_eagerly=False)
    get_one_sample()
    # encode_one()
