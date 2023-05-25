import tensorflow as tf
from tensorflow import keras
from pipeline import TrainPipeline


def mask_feature(feat_start_index, pad_num):
    def func(feature_dict, label):
        print(feature_dict)
        print(label)
        mask_value = [feature_dict["input_value"][:, 0:feat_start_index],
                      tf.zeros_like(feature_dict["input_value"][:, feat_start_index:feat_start_index+pad_num], dtype=tf.float32),
                      feature_dict["input_value"][:, feat_start_index+pad_num:]]
        mask_value = tf.concat(mask_value, axis=1)
        res_feature = {"input_index": feature_dict["input_index"], "input_value": mask_value}
        return res_feature, label
    return func


def feature_evaluation():
    date_time = "2023-05-22 140020"
    model_path = f"../output/{date_time}/model"
    model = keras.models.load_model(model_path)
    result = []

    feat_config_path = f"../output/{date_time}/input_fields.json"
    model_config_path = f"../output/{date_time}/model_config.json"
    pipeline = TrainPipeline(feat_config_path, model_config_path, run_eagerly=False)
    valid_ds = pipeline.read_data(data_type="valid")
    metrics = model.evaluate(valid_ds, return_dict=True)
    result.append(("None", metrics["output_2_auc_1"], 0.0))
    for feat_field, (feat_start_index, pad_num) in pipeline.feat_pad_num_dict.items():
        print(f"======== mask feature: {feat_field} ========")
        valid_ds = pipeline.read_data(data_type="valid")
        valid_ds = valid_ds.map(mask_feature(feat_start_index, pad_num))
        metrics = model.evaluate(valid_ds, return_dict=True)
        result.append((feat_field, metrics["output_2_auc_1"], result[0][1] - metrics["output_2_auc_1"]))
        print(result)
    result.sort(key=lambda x: x[1])
    # 该模型的特征重要度结果，越靠前越重要
    with open(f"../output/{date_time}/feature_evaluation.csv", 'w', encoding='utf-8') as f:
        for tup in result:
            f.write(f"{tup[0]},{tup[1]},{tup[2]}\n")


if __name__ == '__main__':
    feature_evaluation()
