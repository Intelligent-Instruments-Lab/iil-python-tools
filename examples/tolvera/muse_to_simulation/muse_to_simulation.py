import logging
from math import floor
import time

from numpy import clip
import utils
import muse

from pythonosc import udp_client

def send_data_osc(bands, metrics, osc):

    osc.send_message("/bands", bands)
    osc.send_message("/bands/delta", bands[0])
    osc.send_message("/bands/theta", bands[1])
    osc.send_message("/bands/alpha", bands[2])
    osc.send_message("/bands/beta", bands[3])
    osc.send_message("/bands/gamma", bands[4])
    osc.send_message("/metrics/mindfulness", metrics[0])
    osc.send_message("/metrics/restfulness", metrics[1])
    osc.send_message("/metrics", metrics)

def data_to_sim_control(bands, metrics, osc):

    mindfulness = clip(metrics[0], 0.0, 1.0)
    restfulness = clip(metrics[1], 0.0, 1.0)
    delta = clip(bands[0], 0.0, 1.0)
    theta = clip(bands[1], 0.0, 1.0)
    alpha = clip(bands[2], 0.0, 1.0)
    beta = clip(bands[3], 0.0, 1.0)
    gamma = clip(bands[4], 0.0, 1.0)

    boids_radius = round(restfulness * 500)
    boids_speed = mindfulness * 10.0
    boids_cohere = mindfulness
    boids_align = mindfulness
    boids_separate = restfulness
    phys_sense_angle = restfulness * 2.0
    phys_sense_dist = restfulness * 100.0
    phys_move_angle = mindfulness * 2.0
    phys_move_step = mindfulness * 10.0

    osc.send_message(
        "/tolvera",
        [boids_radius,
        boids_speed,
        boids_cohere,
        boids_align,
        boids_separate,
        phys_sense_angle,
        phys_sense_dist,
        phys_move_angle,
        phys_move_step])
    print('/tolvera',
        boids_radius,
        boids_speed,
        boids_cohere,
        boids_align,
        boids_separate,
        phys_sense_angle,
        phys_sense_dist,
        phys_move_angle,
        phys_move_step)

    # osc.send_message("/tolvera/boids/radius", boids_radius)
    # print("/tolvera/boids/radius", boids_radius)

    # osc.send_message("/tolvera/boids/speed", boids_speed)
    # print("/tolvera/boids/speed", boids_speed)

    # osc.send_message("/tolvera/boids/cohere", boids_cohere)
    # print("/tolvera/boids/cohere", boids_cohere)

    # osc.send_message("/tolvera/boids/align", boids_align)
    # print("/tolvera/boids/align", boids_align)

    # osc.send_message("/tolvera/boids/seperate", boids_separate)
    # print("/tolvera/boids/seperate", boids_separate)

    # osc.send_message("/tolvera/physarum/sense_angle", phys_sense_angle)
    # print("/tolvera/physarum/sense_angle", phys_sense_angle)

    # osc.send_message("/tolvera/physarum/sense_dist", phys_sense_dist)
    # print("/tolvera/physarum/sense_dist", phys_sense_dist)

    # osc.send_message("/tolvera/physarum/move_angle", phys_move_angle)
    # print("/tolvera/physarum/move_angle", phys_move_angle)

    # osc.send_message("/tolvera/physarum/move_step", phys_move_step)
    # print("/tolvera/physarum/move_step", phys_move_step)


    """
    OSC input protocol

    /tolvera/boids/radius i 0-500                   detection radius of other boids 
    /tolvera/boids/dt f 0-1                         delta time 
    /tolvera/boids/speed f 0-10                     max speed of boids
    /tolvera/boids/separate f 0-1                   strength of seperation urge
    /tolvera/boids/align f 0-1                      strength of alignment urge
    /tolvera/boids/cohere f 0-1                     strength of wanting to move in the same direction
    /tolvera/physarum/sense_angle f 0-2             angle of sensing field
    /tolvera/physarum/sense_dist f 0-100            sensing distance
    /tolvera/physarum/evaporation f 0.5-0.999       rate of slime evaporation
    /tolvera/physarum/move_angle f 0-2              angle of movement ability
    /tolvera/physarum/move_step f 0-10              speed of movement
    /tolvera/physarum/substep i 1-32                delta time
    """



def update_data(board_shim, osc):

    bands, metrics = muse.get_current_data(board_shim)
    data_to_sim_control(bands, metrics, osc)

def main():

        board_shim = muse.start_session()
        osc = udp_client.SimpleUDPClient("127.0.0.1", 7400)

        print("starting data loop...")
        repeat = utils.RepeatedTimer(0.1, update_data, board_shim, osc)

        try:
            while(1):
                time.sleep(1)
        except BaseException:
            logging.warning('Exception', exc_info=True)
        finally:
            repeat.stop()
            muse.end_session(board_shim)


if __name__ == '__main__':
    main()