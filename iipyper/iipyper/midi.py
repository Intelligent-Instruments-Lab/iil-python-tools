import functools as ft
import time

import mido

from .state import _lock

def _get_filter(item):
    if item is None:
        return item
    if (not isinstance(item, str)) and hasattr(item, '__iter__'):
        return set(item)
    return {item}

class MIDI:
    """"""
    @classmethod
    def print_ports(cls):
        print('Available MIDI inputs:')
        for s in set(mido.get_input_names()):
            print(f'\t{s}')
        print('Available MIDI outputs:')
        for s in set(mido.get_output_names()):
            print(f'\t{s}')
        MIDI.ports_printed = True

    ports_printed = False

    def __init__(self, in_ports=None, out_ports=None, 
                 virtual_in_ports=1, virtual_out_ports=1, 
                 verbose=1, sleep_time=5e-4):
        """
        Args:
            in_ports: list of input devices to open (uses all by default)
            out_ports: list of output devices to open (uses none by default)
            virtual_in_ports: number of 'To iipyper X' ports to create
            virtual_out_ports: number of 'From iipyper X' ports to create
        """
        if not MIDI.ports_printed and verbose:
            MIDI.print_ports()

        self.running = False

        self.verbose = int(verbose)
        self.sleep_time = sleep_time
        # type -> list[Optional[set[port], Optional[set[channel]], function]
        self.handlers = []

        if isinstance(in_ports, str):
            in_ports = in_ports.split(',')
        if isinstance(out_ports, str):
            out_ports = out_ports.split(',')

        # TODO: fuzzy match port names

        if in_ports is None or len(in_ports)==0:
            in_ports = set(mido.get_input_names())

        self.in_ports = {}  
        for i in range(virtual_in_ports):
            virtual_in = f'To iipyper {i+1}'
            self.in_ports[virtual_in] = mido.open_input(
                virtual_in, virtual=True)
        for port in in_ports:
            try:
                self.in_ports[port] = mido.open_input(
                    port, callback=self.get_callback(port))
            except Exception:
                print(f"""WARNING: MIDI input {port} not found""")

        if self.verbose:
            print(f"""opened MIDI input ports: {list(self.in_ports)}""")

        ##### WIP
        self.out_ports = {}
        for i in range(virtual_out_ports):
            virtual_out = f'From iipyper {i+1}'
            self.out_ports[virtual_out] = mido.open_output(
                virtual_out, virtual=True)

        if out_ports is None:
            out_ports = []
        # if out_ports is None or len(out_ports)==0:
            # out_ports = set(mido.get_output_names())  
        # self.out_ports = {}
        for port in out_ports:
            try:
                self.out_ports[port] = mido.open_output(port)
            except Exception:
                print(f"""WARNING: MIDI output {port} not found""")

        if self.verbose:
            print(f"""opened MIDI output ports: {list(self.out_ports)}""")

        self.start()

    def start(self):
        self.running = True

    def handle(self, *a, **kw):
        """MIDI handler decorator
        
        Decorated function receives args:
            msg: mido message
        """
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

    def get_callback(self, port_name):
        # print(port_name)
        def callback(msg):
            if self.verbose > 1:
                print(f'{msg=}')
            if not self.running:
                return
            for filters, f in self.handlers:
                use_handler = (
                    'port' not in filters 
                    or port_name in filters.pop('port'))
                use_handler &= all(
                    filt is None 
                    or not hasattr(msg, k)
                    or getattr(msg, k) in filt
                    for k,filt in filters.items())
                if use_handler:
                    with _lock:
                        print(port_name)
                        f(msg)
        return callback

    def _send_msg(self, port, m):
        """send on a specific port or all output ports"""
        ports = self.out_ports.values() if port is None else [self.out_ports[port]]
        # print(ports)
        for p in ports:
            # print('iipyper send', m)
            # iiuc mido send should already be thread safe
            # with _lock:
            p.send(m)

    # # see https://mido.readthedocs.io/en/latest/message_types.html

    def send(self, m, *a, port=None, **kw):
        """send a mido message"""
        # print(f'SEND {time.perf_counter()}')
        if isinstance(m, mido.Message):
            self._send_msg(port, m)
            if len(a)+len(kw) > 0:
                print('warning: extra arguments to MIDI send')
        elif isinstance(m, str):
            try:
                self._send_msg(port, mido.Message(m, *a, **kw))
            except Exception:
                print('MIDI send failed: bad arguments to mido.Message')
                raise
        else:
            print('MIDI send failed: first argument should be a mido.Message or str')

    def __getattr__(self, name):
        if name=='cc': name = 'control_change'
        if name=='pc': name = 'program_change'
        if name in (
            'note_on', 'note_off', 'polytouch', 'control_change', 
            'program_change', 'aftertouch', 'pitchwheel', 'sysex'):
            return lambda *a, **kw: self.send(name, *a, **kw)
        raise AttributeError
        

