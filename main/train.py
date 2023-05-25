from pipeline import TrainPipeline


def save_model():
    date_time = pipeline.date_time
    model.save(f"../output/{date_time}/model")
    with open(f"../output/{date_time}/input_fields.json", 'w', encoding='utf-8') as f:
        f.write(pipeline.feat_config_str)
    with open(f"../output/{date_time}/model_config.json", 'w', encoding='utf-8') as f:
        f.write(pipeline.model_config_str)
    with open(f"../output/{date_time}/feature_trans", 'w', encoding='utf-8') as f:
        if "all" in pipeline.all_pad_num:
            group = "all"
        elif "user" in pipeline.all_pad_num:
            group = "user"
        else:
            raise ValueError(f"unexpected feature_groups: {pipeline.all_pad_num}")
        f.write(f"{pipeline.all_pad_num[group]}\n")
        for feat_field, (feat_start_index, pad_num) in pipeline.feat_pad_num_dict[group].items():
            f.write(f"{feat_field} {pad_num}\n")
        for feat_field, feat_attr in pipeline.index_dict[group].items():
            for feat_index, global_index in feat_attr.items():
                f.write(f"{feat_field} {feat_index} {global_index}\n")


if __name__ == '__main__':
    feat_config_path = "../config/data_aliyun_down_rec/input_fields.json"
    model_config_path = "../config/data_aliyun_down_rec/model_esmm.json"
    pipeline = TrainPipeline(feat_config_path, model_config_path, run_eagerly=False)
    model = pipeline.train()
    model.summary()
    save_model()
