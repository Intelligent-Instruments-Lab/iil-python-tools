import time
import mido

from iipyper import OSC, MIDI, repeat, run, cleanup

def main(osc_host='127.0.0.1', osc_port=9999, repeat_time=1, repeat_msg='hello'):
    # loop API:
    # use the @repeat decorator to define functions which run every n seconds
    @repeat(repeat_time)
    def _():
        print(f'repeating every {repeat_time} sec: "{repeat_msg}"')

    class timer():
        def __init__(self):
            self.t = time.monotonic_ns()
        def __call__(self):
            new_t = time.monotonic_ns()
            # print(new_t)
            print(f'{1e9/(new_t-self.t)} fps')
            self.t = new_t
    repeat(1/120)(timer())

    ### MIDI API
    # create a MIDI object
    midi = MIDI()
    # NOTE: you probably will want to specify the MIDI outputs and inputs:
    # midi = MIDI(out_ports=['IAC Driver Bus 1'])

    # # decorator to make a midi handler:
    # # here filtering for type='note_on', note > 0, channel = 0
    @midi.handle(type='note_on', note=range(1,128), channel=0)
    def _(msg):
        print(msg)
        # time.sleep(0.2)
        # print('end of sleep\n')

        # 200_000 ns = 200 us = .2 ms
        # 60fps = ~15ms

    @midi.handle(type='control_change')
    def _(msg):
        print(msg)

    # function to send MIDI:
    midi.send('note_on', note=60, velocity=100, channel=0)
    # or mido-style:
    m = mido.Message('note_off', note=60, velocity=100, channel=0)
    midi.send(m)

    @repeat(1)
    def _():
        # print('sending note on')
        midi.note_on(note=60, velocity=100, channel=0)
        # print('sending cc')
        midi.cc(control=0, value=127, channel=1)

    # NOTE: by default, iipyper sends and receives on all MIDI ports -- 
    # so here the handler prints every MIDI message the @repeat function sends

    # make an OSC object
    osc = OSC(osc_host, osc_port)
    # # OSC API:
    # # osc.args, osc.kwargs

    # # use as a decorator to make an osc handler

    # # positional arguments
    # # osc route is taken from function name
    @osc.args
    def test(address, x, y): # route is /test
        print(address, x, y)
        # if a value (besides None) is returned,
        # it will be sent back as an OSC message to the sender.
        # you don't have to specify the IP or port of the client this way.
        # the first element should be the OSC address:
        return '/test', x-y, x+y

    # the above works with SuperCollider, but not with Max
    # the reply port Max sends doesn't seem to work conveniently.
    # (you can't make a new udpsend object with that port, it's taken)
    # but you can specify your own return port like this:
    @osc.args(return_port=5432)
    def test2(address, x, y): # route is /test2
        print(address, x, y)
        return '/test', x-y, x+y # will send on port 5432

    # named arguments as key, value pairs in OSC
    # e.g. an OSC message ["/keyword_example", "arg2", 3]
    # would print "/keyword_example 0 3 99"
    @osc.kwargs
    def keyword_example(address, arg1=0, arg2=1, arg3=99):
        print(address, arg1, arg2, arg3)
        # no return value: does not send OSC back

    # you can also give the OSC address explicitly to the decorator,
    # instead of using the function name.
    # this supports wildcards and other aspects of OSC addresses:
    @osc.args('/math/*')
    def _(address, a, b):
        print(address, a, b)
        op = address.split('/')[-1]
        if op=='add':
            return address, a + b
        if op=='mul':
            return address, a * b

    # OSC clients can be created explicitly and given names:
    osc.create_client('supercollider', port=57120) # uses same host as server
    # but clients will also be created automatically when possible

    # # to send osc:
    osc.send('/default_send_test', 0) # send to default (first created) client

    # # send with client in the route:
    osc.send('127.0.0.1:57120/other_send_test', 1) 

    # # send to named client:
    osc.create_client('supercollider2', port=57121)
    osc.send('/other_send_test', 2, client='supercollider2')
    # # alternate send syntax:
    osc('supercollider2', '/other_send_test', 3)

    # functions with the @cleanup decorator will run before exit on KeyboardInterrupt
    @cleanup
    def _():
        print('exiting...')
        osc.send('/default_send_test', 'bye')


if __name__=='__main__':
    run(main)