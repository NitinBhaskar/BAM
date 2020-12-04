#!/bin/bash

# activate the conda virtual environment
conda activate vitis-ai-tensorflow

# folders
export OUT=./out
export CHKPT_FILENAME='chkpt.ckpt'
export GRAPH_NAME='model.pb'
export FROZEN_GRAPH=frozen_model.pb
export TARGET_DIR=./target
export ARCH=/workspace/ULTRA96V2.json

export INPUT_HEIGHT=32
export INPUT_WIDTH=16
export INPUT_CHAN=1
export INPUT_SHAPE=?,${INPUT_HEIGHT},${INPUT_WIDTH},${INPUT_CHAN}
export INPUT_NODE=conv2d_input
export OUTPUT_NODE=dense/BiasAdd
export NET_NAME=bam

mkdir -p ${OUT}

echo "****************************** training *********************************"
# Start the training
python train_keras.py


echo "****************************** freezing *********************************"
python keras_2_tf.py --keras_hdf5 ${OUT}/keras_out.h5 \
                     --tf_ckpt=${OUT}/tf_chkpt.ckpt

freeze_graph \
    --input_meta_graph  ${OUT}/tf_chkpt.ckpt.meta \
    --input_checkpoint  ${OUT}/tf_chkpt.ckpt \
    --output_graph      ${OUT}/${FROZEN_GRAPH} \
    --output_node_names ${OUTPUT_NODE} \
    --input_binary      true


echo "****************************** quantizing *********************************"
# Quantize
vai_q_tensorflow quantize \
	--input_frozen_graph ${OUT}/${FROZEN_GRAPH} \
	--input_fn           image_input_fn.calib_input \
	--output_dir         ${OUT} \
	--input_nodes        ${INPUT_NODE} \
	--output_nodes       ${OUTPUT_NODE} \
	--input_shapes       ${INPUT_SHAPE} \
	--calib_iter         2 \
	--gpu                "0"

# Need below command only once
#dlet -f ULTRA96V2.hwh

echo "****************************** compiling *********************************"
# Compile the model
vai_c_tensorflow \
    --frozen_pb  ${OUT}/deploy_model.pb \
    --arch       ${ARCH} \
    --output_dir ${OUT} \
    --net_name   ${NET_NAME}

echo "****************************** copying *********************************"
# Copy the elf file to target folder
cp ${OUT}/*.elf ${TARGET_DIR}
cp ./*.npy ${TARGET_DIR}
