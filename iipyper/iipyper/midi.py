import asyncio

import mido

# not sure why this didn't work in MIDI class.
async def midi_coroutine(self):
    while True:
        for port_name, port in self.in_ports.items():
            # print(port_name, port)
            for m in port.iter_pending():
                # print(port_name, m)
                for filters, f in self.handlers:
                    use_handler = (
                        'port' not in filters or port_name in filters.pop('port'))
                    use_handler &= all(
                        filt is None 
                        or not hasattr(m, k)
                        or getattr(m, k) in filt
                        for k,filt in filters.items())
                    if use_handler: f(m)
                    # print([(
                    #     filt is None,
                    #     not hasattr(m, k),
                    #     getattr(m, k) in filt, k, filt)
                    #     for k,filt in filters.items()])

        await asyncio.sleep(self.sleep_time)

def _get_filter(item):
    if item is None:
        return item
    if (not isinstance(item, str)) and hasattr(item, '__iter__'):
        return set(item)
    return {item}

class MIDI:
    """"""
    instances = []
    def __init__(self, in_ports=None, out_ports=None, verbose=True, sleep_time=0.0005):
        """
        Args:
            in_ports: list of input devices (uses all by default)
            out_ports: list of output devices (uses all by default)
        """
        self.verbose = verbose
        self.sleep_time = sleep_time
        # type -> list[Optional[set[port], Optional[set[channel]], function]
        self.handlers = []

        if in_ports is None or len(in_ports)==0:
            in_ports = mido.get_input_names()  
        self.in_ports = {# mido.ports.MultiPort([
            port: mido.open_input(port)#, callback=self.get_midi_callback())
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

        # self.handle = MIDIHandlers(self)
        MIDI.instances.append(self)

    def handle(self, *a, **kw):
        """MIDI handler decorator"""
        if len(a):
            # bare decorator
            assert len(a)==1
            assert len(kw)==0
            assert hasattr(a[0], '__call__')
            f = a[0]
            filters = {}
        else:
            # with filter arguments
            for k in kw:
                assert k in {
                    'channel', 'port', 'type', 
                    'note', 'velocity', 'value', 
                    'control', 'program'
                    }, f'unknown MIDI message filter "{k}"'
            filters = {k:_get_filter(v) for k,v in kw.items()}
            f = None

        def decorator(f):
            self.handlers.append((filters, f))
            return f
        
        return decorator if f is None else decorator(f)

    def _send_msg(self, port, m):
        """send on a specific port or all output ports"""
        ports = self.out_ports.values() if port is None else [self.out_ports[port]]
        for p in ports:
            p.send(m)

    # # see https://mido.readthedocs.io/en/latest/message_types.html

    def send(self, m, *a, port=None, **kw):
        """send a mido message"""
        if isinstance(m, mido.Message):
            self._send_msg(port, m)
            if len(a)+len(kw) > 0:
                print('warning: extra arguments to MIDI send')
        elif isinstance(m, str):
            try:
                self._send_msg(port, mido.Message(m, *a, **kw))
            except Exception:
                print('MIDI send failed: bad arguments to mido.Message')
        else:
            print('MIDI send failed: first argument should be a mido.Message or str')

