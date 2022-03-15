from collections import defaultdict

import mido

def _get_filter(item):
    if item is None:
        return item
    if hasattr(item, '__iter__'):
        return set(item)
    return {item}

class MIDI:
    """"""
    def __init__(self, in_ports=None, out_ports=None, verbose=True):
        """
        Args:
            in_ports: list of input devices (uses all by default)
            out_ports: list of output devices (uses all by default)
        """
        self.verbose = verbose
        # type -> list[Optional[set[port], Optional[set[channel]], function]
        self.handlers = defaultdict(list)

        if in_ports is None or len(in_ports)==0:
            in_ports = mido.get_input_names()  
        self.in_ports = {# mido.ports.MultiPort([
            port: mido.open_input(port, callback=self.get_midi_callback())
            for port in in_ports
        }

        if self.verbose:
            print(f"""opened MIDI input ports: {list(self.in_ports)}""")

        if out_ports is None or len(out_ports)==0:
            out_ports = mido.get_output_names()  
        self.out_ports = {# mido.ports.MultiPort([
            port: mido.open_output(port)
            for port in out_ports
        }

        if self.verbose:
            print(f"""opened MIDI output ports: {list(self.out_ports)}""")

        self.handle = MIDIHandlers(self)

    # TODO: filtering needs to work for all message types...
    # maybe there should be just one decorator
    # which can accept any filter, including on type?
    # and which just takes a mido message?

    def get_midi_callback(self):
        """callback for mido MIDI handling"""
        # close over `self``
        def callback(m):
            """dispatch to decorated midi handlers"""
            # logging.debug(m)
            handlers = self.handlers[m.type]
            for ports, channels, numbers, values, f in handlers:
                use_handler = (
                    (ports is None or m.port in ports) and
                    (channels is None or m.channel in channels) and
                    (numbers is None or m.number in numbers) and
                    (values is None or m.value in values) 
                )
                if use_handler:
                    f(m)
        return callback

    def _decorator(self, msg_type, 
            ports=None, channels=None, numbers=None, values=None):
        """generic MIDI handler decorator"""
        if hasattr(ports, '__call__'):
            # bare decorator
            f = ports
            assert channels is None
            assert numbers is None
            assert values is None
        else:
            # with filter arguments
            f = None
            ports = _get_filter(ports)
            channels = _get_filter(channels)
            numbers = _get_filter(numbers)
            values = _get_filter(values)

        def decorator(f):
            self.handlers[msg_type].append([
                ports, channels, numbers, values, f
            ])
            return f
        
        return decorator if f is None else decorator(f)

    # send on all ports by default
    def _send(self, port, *a, **kw):
        ports = self.out_ports.values() if port is None else [self.out_ports[port]]
        for p in ports:
            p.send(mido.Message(*a, **kw))

    # see https://mido.readthedocs.io/en/latest/message_types.html

    def note_on(self, note, velocity=64, channel=0, port=None):
        self._send(port, 'note_on', channel=channel, note=note, velocity=velocity)

    def note_off(self, note, velocity=64, channel=0, port=None):
        self._send(port, 'note_off', channel=channel, note=note, velocity=velocity)

    def control_change(self, control, value, channel=0, port=None):
        self._send(port, 'control_change', channel=channel, control=control, value=value)

    def cc(self, *a, **kw):
        self.control_change(*a, **kw)

    def program_change(self, program, channel=0, port=None):
        self._send(port, 'program_change', channel=channel, program=program)

    def polytouch(self, note, value=64, channel=0, port=None):
        self._send(port, 'polytouch', channel=channel, note=note, value=value)

    def aftertouch(self, value=64, channel=0, port=None):
        self._send(port, 'aftertouch', channel=channel, value=value)

    def pitchwheel(self, pitch=0, port=None):
        self._send(port, 'pitchwheel', pitch=pitch)

    def pitchbend(self, *a, **kw):
        self.pitchwheel(*a, **kw)


# this is effectively part of MIDI class, 
# it is only a separate class for naming aesthetics
class MIDIHandlers:
    """specific MIDI handler decorators"""
    def __init__(self, midi):
        self.midi = midi

    def note_on(self, ports=None, channels=None, notes=None, velocities=None):
        return self.midi._decorator('note_on', ports, channels, notes, velocities)

    def note_off(self, ports=None, channels=None, notes=None, velocities=None):
        return self.midi._decorator('note_off', ports, channels, notes, velocities)

    def control_change(self, ports=None, channels=None, controls=None, values=None):
        return self.midi._decorator('control_change', ports, channels, controls, values)

    def cc(self, *a, **kw):
        return self.control_change(*a, **kw)
