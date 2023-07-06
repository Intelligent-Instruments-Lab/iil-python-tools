"""
This example demonstrates how to use the OSCMap class to map OSC messages to Python functions and vice versa.

When this example is run, a Pd patch is created in the background.
When the Pd patch is opened, it will send and receive messages to this iipyper server.
It is possible to output a Max patch instead by altering the command-line arguments.
"""

from iipyper import OSC, OSCMap, run, repeat

def main(host="127.0.0.1", client="127.0.0.1", receive_port=5001, send_port=5000, # osc
         patch_type="Pd", patch_name="iipyper-oscmap", # osc mapping
        ):
    
    '''''''''''''''''''''''''''''''''''''''''''''
    OSC

    First, we create an OSC instance and a client. 
    '''''''''''''''''''''''''''''''''''''''''''''
    osc = OSC(host, receive_port, verbose=True, concurrent=True)
    client_name = "iipyper-example"
    osc.create_client(client_name, client, send_port)
    
    '''''''''''''''''''''''''''''''''''''''''''''
    OSC Mapping

    Then, we create an OSCMap instance.
    Next, we add functions to the OSCMap instance using the `add` method.
    In the `add` method we specify OSC parameters like this: `name=(default, min, max)`.
    The example OSCMap shows one send function and one receive function.
    '''''''''''''''''''''''''''''''''''''''''''''
    patch_dir = "examples/iipyper/" if patch_type=="Pd" else "max"
    appendix = "_local" if host=="127.0.0.1" else "_remote"
    patch_filepath = patch_dir+'/'+patch_name+appendix
    osc_map = OSCMap(osc, client_name, patch_type, patch_filepath)

    '''
    Patcher → Python
    '''
    io, update_rate = 'receive', 5
    receive_counter = 0

    # Receive test `/from/pd`
    @osc_map.add(f_val=(0.5,0,1), io='receive', count=update_rate)
    def from_pd(f_val: float):
        nonlocal receive_counter
        receive_counter += f_val
        print(f"Received /from/pd {f_val} and added to `receive_counter`: {receive_counter}")

    '''
    Python → Patcher
    '''
    io, update_rate = 'send', 7
    send_mode = 'broadcast' # | 'event'
    send_counter = 0

    # Receive test
    @osc_map.add(i_val=(50,0,100), io=io, count=update_rate, send_mode=send_mode)
    def to_pd(i_val: int=100) -> tuple[str]:
        nonlocal send_counter
        send_counter += 1
        print(f"Sending /to/pd {send_counter}")
        return [send_counter]

    @repeat(0.125)
    def _():
        osc_map()

if __name__=='__main__':
    run(main)
