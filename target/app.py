import os
import time
import numpy as np
import math
import argparse
import threading
import sys
import time 

from queue import Queue
from serial import Serial
from mindlink import read_raw_eeg

from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.set_runtime("vart")
overlay.load_model("dpu_bam.elf") # Compiled model

# Read out data from mindlink
def producer(out_q, ser, common_q):
    total_run = common_q.get()
    out_q.put(total_run)
    while total_run > 0:
        samples = read_raw_eeg(ser, 512) # Fetch 1 second of reading
        # Put the samples in the queue for consumer to fetch
        out_q.put(samples)
        total_run -= 1
        #print('Producer')


def consumer(in_q, common_q):
    med = 0
    dpu = overlay.runner # DPU
    # Tensors
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    tensorformat = dpu.get_tensor_format()
    # Tensors output shape
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

    # Input shape
    shape_in = (1,) + tuple(
        [inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])
    # Output shape
    shape_out = (1, outputHeight, outputWidth, outputChannel)

    # Loop till total_run posted by producer thread
    total_run = in_q.get()
    while total_run > 0:
        total_run -= 1
        q_data = in_q.get()
        #print('Consumer')
        input_data = []
        output_data = []
        input_data.append(np.empty((shape_in), 
                                dtype = np.float32, order = 'C'))
        output_data.append(np.empty((shape_out), 
                                dtype = np.float32, order = 'C'))
        inData = input_data[0]

        inData[0,...] = q_data.reshape(
            inputTensors[0].dims[1],
            inputTensors[0].dims[2],
            inputTensors[0].dims[3])
        # DPU runs input through the graph
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        # If 1 then meditating probability is higher
        if np.argmax(output_data) == 1:
            med += 1
            sys.stdout.write("meditating\r")
        else:
            sys.stdout.write("distracted\r")

    common_q.put(med) # Put the med value in to common queue

def app(model, com_port, t_in_sec):
    q = Queue() # Queue to share EEF raw data
    common_q = Queue() # Queue to sync between main, producer and consumer threads
    time_period = int(t_in_sec) # Time to run in seconds
    ser = Serial(com_port, 57600, timeout=None) # Serial port init

    t1 = threading.Thread(target = producer, args =(q, ser, common_q)) # Init thread1
    t1.start() # Start the thread
    t2 = threading.Thread(target = consumer, args =(q, common_q)) # Init thread2
    t2.start() # Start the thread

    # Start producer and consumer for t_in_sec seconds
    common_q.put(int(t_in_sec))

    # Wait till time_period
    time.sleep(int(t_in_sec))

    # Wait till consumer thread consumes 'Done' message and puts meditation level in common queue    
    time.sleep(2)
    out = common_q.get() # Read meditation level value
    print('Meditation level was %.2f%%' %(out/int(t_in_sec)*100))

    t1.join()
    t2.join()
    

def main():

  ap = argparse.ArgumentParser()  
  ap.add_argument('-m', '--model',
                  type=str,
                  default='./dpu_bam.elf',
                  help='Path of .elf. Default is model_dir/dpu_customcnn.elf')
  ap.add_argument('-s', '--serial',
                  type=str,
                  default='/dev/ttyS0',
                  help='Serial port device node')
  ap.add_argument('-t', '--time',
                  type=str,
                  default='20',
                  help='Time in seconds')

  args = ap.parse_args()
  
  print ('Command line options:')
  print (' --model     : ', args.model)
  print (' --serial   : ', args.serial)
  print (' --time   : ', args.time)

  app(args.model, args.serial, args.time)

if __name__ == '__main__':
  main()

