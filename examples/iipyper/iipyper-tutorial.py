import mido

from iipyper import OSC, MIDI, repeat, run

# TODO: MIDI
# TODO: loops are broken, why?

def main(osc_host='127.0.0.1', osc_port=9999, loop_time=1, loop_msg='hello'):
    # loop API:
    # use the @repeat decorator to define functions which run every n seconds
    @repeat(loop_time)
    def _():
        print(f'looping every {loop_time} sec: "{loop_msg}"')

    @repeat(1.5)
    def _():
        print('looping every 1.5 sec')

    midi = MIDI()
    # # MIDI API:
    # # midi.note_on, midi.cc, etc

    # # decorator to make a midi handler:
    # # here filtering for type='note_on', note > 0, channel = 0
    @midi.handle(type='note_on', note=range(1,128), channel=0)
    def _(msg):
        print(msg)

    # function to send MIDI:
    midi.send('note_on', note=60, velocity=100, channel=0)
    # or mido-style:
    m = mido.Message('note_off', note=60, velocity=100, channel=0)
    midi.send(m)

    # @repeat(1)
    # def _():
    #     midi.note_on(pitch=60, velocity=100, channel=0)
    #     midi.cc(number=0, value=127, channel=1)


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
        # it will be sent back as an OSC message to the sender
        # with the route given as the first element:
        return '/test', x-y, x+y

    # named arguments as key, value pairs in OSC
    # e.g. an OSC message ["/keyword_example", "arg2", 3]
    # would print "/keyword_example 0 3 99"
    @osc.kwargs
    def keyword_example(address, arg1=0, arg2=1, arg3=99):
        print(address, arg1, arg2, arg3)
        # no return value: does not send OSC back

    # can also give the route explictly to the decorator,
    # supporting wildcards
    @osc.args('/math/*')
    def _(address, a, b):
        print(address, a, b)
        op = address.split('/')[-1]
        if op=='add':
            return address, a + b
        if op=='mul':
            return address, a * b

    # OSC clients can be created explcitly and given names:
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

# it may be possible to have async/threaded option for both OSC and MIDI?
# MIDI is threaded and OSC is async
# but MIDI could be enqueued and handled in the loop,
# OSC could launch the threading server?

if __name__=='__main__':
    run(main)