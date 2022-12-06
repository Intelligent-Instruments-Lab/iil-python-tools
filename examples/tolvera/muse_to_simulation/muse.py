import argparse
import logging
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

def get_current_data(board_shim):
    """
    gets the current data from the given board
    format: [delta, theta, alpha, beta, gamma], [mindfulness, restfulness]
    """
    board_id = board_shim.get_board_id()
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    window_size = 4
    num_points = window_size * sampling_rate
    data = board_shim.get_current_board_data(num_points)

    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    feature_vector = bands[0]

    print(feature_vector)

    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    mindfulness_prediction = mindfulness.predict(feature_vector)
    print('Mindfulness: %s' % str(mindfulness_prediction))
    mindfulness.release()
    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    restfulness_prediction = restfulness.predict(feature_vector)
    print('Restfulness: %s' % str(restfulness_prediction))
    restfulness.release()

    metrics_vector = (float(mindfulness_prediction), float(restfulness_prediction))

    return feature_vector, metrics_vector

def start_session():
    """
    prepares, starts and returns the board
    """
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    parser.add_argument('--preset', type=int, help='preset for streaming and playback boards',
                        required=False, default=BrainFlowPresets.DEFAULT_PRESET)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board
    params.preset = args.preset

    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
        # if the board_shim is returne imediately, the update function is called, bevore the board is ready
        # this was only tested with the synthetic board
        time.sleep(0.2)
        return board_shim

    except BaseException:
        logging.warning('Exception', exc_info=True)

def end_session(board_shim):
    """
    releases the board
    """
    logging.info('End')
    if board_shim.is_prepared():
        logging.info('Releasing session')
        board_shim.release_session()