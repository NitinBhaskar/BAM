from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.set_runtime("vart")
overlay.load_model("dpu_bam.elf")

DATASET_X_FILENAME='x.npy'
DATASET_Y_FILENAME='y.npy'

import os
import time
import numpy as np
import math


def calculate_softmax(data, size):
    sum=0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum +=result[i]
    for i in range(size):
        result[i] /=sum
    return result

def app():
    # dataset
    x_data = np.load(DATASET_X_FILENAME)
    y_data = np.load(DATASET_Y_FILENAME)
    
    labels = []
    for y in y_data:
        if y == 0:
            labels.append([1, 0])
        else:
            labels.append([0, 1])

    correct = 0
    wrong = 0
    for i in range(len(x_data)):
        res = run(x_data[i])

        if res == y_data[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct/(correct + wrong)
    print('Correct:',correct,'Wrong:',wrong,'Accuracy:', accuracy)

def run(in_q):
    dpu = overlay.runner
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    tensorformat = dpu.get_tensor_format()
    if tensorformat == dpu.TensorFormat.NCHW:
        outputHeight = outputTensors[0].dims[2]
        outputWidth = outputTensors[0].dims[3]
        outputChannel = outputTensors[0].dims[1]
    elif tensorformat == dpu.TensorFormat.NHWC:
        outputHeight = outputTensors[0].dims[1]
        outputWidth = outputTensors[0].dims[2]
        outputChannel = outputTensors[0].dims[3]
    else:
        raise ValueError("Image format error.")
    outputSize = outputHeight*outputWidth*outputChannel


    shape_in = (1,) + tuple(
        [inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])
    shape_out = (1, outputHeight, outputWidth, outputChannel)
    input_data = []
    output_data = []
    input_data.append(np.empty((shape_in), 
                                dtype = np.float32, order = 'C'))
    output_data.append(np.empty((shape_out), 
                                dtype = np.float32, order = 'C'))
    inData = input_data[0]

    inData[0,...] = in_q.reshape(
        inputTensors[0].dims[1],
        inputTensors[0].dims[2],
        inputTensors[0].dims[3])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    return np.argmax(output_data)


# only used if script is run as 'main' from command line
def main():

  app()

if __name__ == '__main__':
  main()
