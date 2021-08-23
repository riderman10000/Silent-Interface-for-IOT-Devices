import requests
import numpy as np 
import time
import tqdm

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

from playsound import playsound

from emg_lib import signal_pipeline, parser

# URL = 'http://127.0.0.1:5000/takeEmg'
URL = "https://silent-app-test.herokuapp.com/takeEmg"
PORT = '/dev/ttyUSB0'
SESSION_RECORD_INTERVAL = 5 # seconds 

# initalizing the board
BoardShim.enable_dev_board_logger()

boardParameters = BrainFlowInputParams()
boardParameters.serial_port = PORT 

board_id = BoardIds.SYNTHETIC_BOARD.value #BoardIds.CYTON_BOARD.value #
board = BoardShim(board_id, boardParameters)
channels = board.get_emg_channels(board_id)

print("Starting the recording.")
print("Preparing session, speak on beep")
time.sleep(1)

playsound('./resources/beep.mp3')

board.prepare_session()
board.start_stream()
BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')

for i in zip(tqdm.tqdm(range(SESSION_RECORD_INTERVAL),desc="RECORDING DATA : ")):
    time.sleep(1)

data = board.get_board_data()
board.stop_stream()
board.release_session()

playsound('./resources/beep.mp3')

DataFilter.write_file(data, 'recording/currentRecording.csv', 'w')
data = np.transpose(data)
channel_data = data[:, channels]

# to test with data...
import glob
import os 
MAIN_DIR = "."
while os.path.basename(os.getcwd())!="Silent-Interface-for-IOT-Devices":
    os.chdir("..")


testfiles = glob.glob('./**/test/1.txt', recursive=True)
print(testfiles)
all_files = testfiles

rawdata = parser(all_files, NORMALIZE=True, utteranceCountPerFile = 5)

all_data_filtered = rawdata.copy()
all_data_filtered['data'] = signal_pipeline(rawdata['data'])

# Todo : need to decide if processing should be done in this script or flask
# TODO : need to trim the length of the recording equal to trained data length
filteredData = all_data_filtered['data'][0]

# emgData = np.array([1,2,3,4,5,6])
# dataPayload = {'emg' : filteredData.tolist()}
# r = requests.post(URL, json=dataPayload)


resetPayload = {'reset' : 'resets the status in server hopefully'}
r = requests.post(URL, json=resetPayload)