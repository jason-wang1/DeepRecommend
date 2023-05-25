# DeepRec
DeepRec 是一个易用、易扩展、模块化的深度学习推荐算法项目，采用 `TensorFlow 2`实现

## 环境要求
* TensorFlow 2.x
* Python 3.8

## 实现的模型
| 模型   |  描述   |
| ---- | ---- |
|   TwoTowerDeepFM   |   深度向量双塔模型，并把FM双塔化，实现了user塔与item塔的显示特征交叉   |
|   [ESMM](https://arxiv.org/abs/1804.07931)   |   通过预估CTR、CTCVR来间接预估CVR，缓解传统CVR预估的样本选择偏差、样本稀疏问题   |

## 特征输入编码规则
特征编码为dict结构：{"input_index": index_tensor, "input_value": value_tensor}
* index_tensor shape: [batch_size, all_pad_num]
* index_tensor dtype: tf.int32
* input_value shape: [batch_size, all_pad_num]
* input_value dtype: tf.float32
* all_pad_num为所有特征所占用坑位数量
