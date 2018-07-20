# -*- coding: utf-8 -*-
# @Time    : 18-7-20 上午10:50
# @Author  : Yan
# @Site    : 
# @File    : tool_k2tf_model_convert.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:

from keras import backend as K
import tensorflow as tf
from keras.models import load_model

K.set_learning_phase(1)

model = load_model('./model/20180528/sys_king_res34_e50d5.h5')

print(model.input)
print(model.output)

legacy_init_op = tf.group(tf.tables_initializer())

with K.get_session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())

    export_path = './model/20180528/k2tf_model/sys_king_res34_e50d5'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    signature_inputs = {
        'main_input': tf.saved_model.utils.build_tensor_info(model.input),
        # 'feature_input': tf.saved_model.utils.build_tensor_info(model.input[1]),
    }

    signature_outputs = {
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: tf.saved_model.utils.build_tensor_info(model.output)
    }

    classification_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=signature_inputs,
        outputs=signature_outputs,
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_webshell_php': classification_signature_def
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()
