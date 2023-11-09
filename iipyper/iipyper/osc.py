from typing import Tuple
import time
import json
from threading import Thread

from pythonosc import osc_packet
# from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.osc_server import BlockingOSCUDPServer, ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

from .state import _lock

# leaving this here for now. seems like it may not be useful since nested bundles
# do not appear to work in sclang.
# class BundleDispatcher(Dispatcher):
#     def call_handlers_for_packet(self,
#              data: bytes, client_address: Tuple[str, int]
#              ) -> None:
#         """Override python-osc to handle whole bundles in one callback.
#             data: Data of packet
#             client_address: Address of client this packet originated from
#         """
#         # Get OSC messages from all bundles or standalone message.
#         try:
#             packet = osc_packet.OscPacket(data)
#             addrs = [msg.message.address for msg in packet.messages]
#             # get common root address
#             root_addr = '/root' # TODO
#             stem_addrs = addrs # TODO
#             handler = self.handlers_for_address(root_addr)
#             # TODO: handle time
#             # times = [msg.time for msg in packet.messages]
#             fused_message = []
#             for stem, msg in zip(stem_addrs, packet.messages):
#                 fused_message.append(stem)
#                 fused_message.append(msg.message)
#             handler.invoke(client_address, fused_message)
#         except osc_packet.ParseError:
#             pass

def do_json(d, k, json_keys, route):
    v = d[k]
    if not isinstance(v, str): return
    has_prefix = v.startswith('%JSON:')
    if has_prefix:
        v = v[6:]
    if k in json_keys or has_prefix:
        try:
            d[k] = json.loads(v)
        except (TypeError, json.JSONDecodeError) as e:
            print(f"""
            warning: JSON decode failed for {route} argument "{k}": 
            value: {v}
            {type(e)} {e}
            """)

class OSC():
    """
    TODO: Handshake between server and clients
    TODO: Polling clients after handshake
    TODO: Enqueuing and buffering messages (?)
    """
    def __init__(self, host="127.0.0.1", port=9999, verbose=True,
         concurrent=False):
        """
        TODO: Expand to support multiple IPs + ports

        Args:
            host (str): IP address
            port (int): port to receive on
            verbose (bool): whether to print activity
            concurrent (bool): if True, handle each incoming OSC message on 
                its own thread. otherwise, incoming OSC is handled serially on 
                one thread for the whole OSC object.
        """
        self.verbose = verbose
        self.concurrent = concurrent
        self.host = host
        self.port = port
        self.dispatcher = Dispatcher()
        self.server = None
        self.clients = {} # (host,port) -> client
        self.client_names = {} # (name) -> (host,port)

        self.create_server()

    def create_server(self):#, host=None, port=None):
        """
        Create the server
        """
        # if (host is None):
        #     host = self.host
        # if (port is None):
        #     port = self.port
        cls = ThreadingOSCUDPServer if self.concurrent else BlockingOSCUDPServer

        if (self.server is None):
            self.server = cls((self.host, self.port), self.dispatcher)
            if self.verbose:
                print(f"OSC server created {self.host}:{self.port}")

            # start the OSC server on its own thread
            Thread(target=self.server.serve_forever, daemon=True).start()
            # self.server.serve_forever()
        else:
            print("OSC server already exists")

    # def close_server(self):
    #     """
    #     Close the server
    #     """
    #     if (self.server is not None):
    #         self.transport.close()
    #     else:
    #         print("OSC server does not exist")

    def add_handler(self, address, handler):
        """
        Map the custom message handler to the OSC dispatcher
        """
        # if (self.server is not None):
        self.dispatcher.map(address, handler, needs_reply_address=True)

    def create_client(self, name, host=None, port=None):
        """
        Add an OSC client.
        Args:
            name: name this client
            host (int): IP to send to, defaults to same as server
            port (int): port to send to, defaults to 57120 (supercollider)
        """
        if (host == None):
            host = self.host
        if (port == None):
            port = 57120
        if ((host, port) not in self.clients):
            self.clients[host, port] = SimpleUDPClient(host, port)
            if self.verbose:
                print(f"OSC client created {host}:{port}")
        else:
            print("OSC client already exists")
        self.client_names[name] = (host, port)

    def get_client_by_name(self, name):
        try:
            return self.clients[self.client_names[name]]
        except Exception:
            print(f'no client with name "{name}"')
            return None

    def get_client_by_sender(self, address):
        if address not in self.clients:
            host, port = address
            self.create_client(f'{host}:{port}', host, port)
        return self.clients[address]

    def send(self, route, *msg, client=None):
        """
        Send message to default client, or with client in address

        Args:
            address: '/my/osc/route' or 'host:port/my/osc/route'
            *msg: content
            client: name of client or None
        """
        if client is not None:
            client = self.get_client_by_name(client)
        elif ':' in route:
            try:
                client_str, route = route.split('/', 1)
                assert ':' in client_str
                host, port = client_str.split(':')
                assert '/' not in host
                port = int(port)
                client = self.get_client_by_sender((host, port))
            except Exception:
                print(f'failed to get client address from OSC route "{route}"')
        else:
            client = next(iter(self.clients.values()))

        if client is None:
            print(f'OSC message failed to send, could not determine client')
            return

        if not route.startswith('/'):
            route = '/'+route
        client.send_message(route, msg)
        if self.verbose:
            print(f"OSC message sent {route}:{msg}")

    def _decorate(self, use_kwargs, route, return_host, return_port, json_keys):
        """generic decorator (args and kwargs cases)"""
        if hasattr(route, '__call__'):
            # bare decorator
            f = route
            route = None
            json_keys = set()
        else:
            f = None
            json_keys = set(json_keys or [])

        def decorator(f, route=route, 
                return_host=return_host, return_port=return_port, 
                json_keys=json_keys):
            # default_route = f'/{f.__name__}/*'
            if route is None:
                route = f'/{f.__name__}'
            # print(route)
            assert isinstance(route, str) and route.startswith('/')

            def handler(client, address, *args):
                """
                Args:
                    client: (host,port) of sender
                    address: full OSC address
                    *args: content of OSC message
                """
                # print('handler:', client, address)
                if use_kwargs:
                    kwargs = {k:v for k,v in zip(args[::2], args[1::2])}
                    # JSON conversions
                    for k in kwargs: 
                        do_json(kwargs, k, json_keys, route)
                    args = []
                else:
                    kwargs = {}

                with _lock:
                    ret = f(address, *args, **kwargs)
                if ret is not None:
                    self.return_to_sender_by_sender(ret, client, return_host, return_port)

            self.add_handler(route, handler)

            return f

        return decorator if f is None else decorator(f)
    
    def return_to_sender_by_sender(self, return_to: tuple, sender: tuple, return_host=None, return_port=None):
        '''
        Args:
            return_to: tuple of (route, *args) to send back to sender
            client: (host,port) of sender
            return_host: host to send back to, defaults to sender
            return_port: port to send back to, defaults to sender
        
        Send return value as message back to sender by client(host,port).
        '''
        if not hasattr(return_to, '__len__'):
            print(f"""
            value returned from OSC handler should start with route, got: {return_to}
            """)
        else:
            sender = (
                sender[0] if return_host is None else return_host,
                sender[1] if return_port is None else return_port
            )
            self.get_client_by_sender(sender).send_message(return_to[0], return_to[1:])

    def return_to_sender_by_name(self, return_to: tuple, client_name: str):
        '''
        Args:
            return_to: tuple of (route, *args) to send back to sender
            client_name: name of client
        
        Send return value as message back to sender by name.
        '''
        if not hasattr(return_to, '__len__'):
            print(f"""
            value returned from OSC handler should start with route, got: {return_to}
            """)
        else:
            self.get_client_by_name(client_name).send_message(return_to[0], return_to[1:])

    def args(self, route=None, return_host=None, return_port=None):
        """decorate a function as an args-style OSC handler.

        the decorated function should look like:
        def f(route, my_arg, my_arg2, ...):
            ...
        the OSC message will be converted to python types and passed as positional
        arguments.
        """
        return self._decorate(False, route, return_host, return_port, None)

    def kwargs(self, route=None, return_host=None, return_port=None, json_keys=None):
        """decorate a function as an kwargs-style OSC handler.

        the decorated function should look like:
        def f(route, my_key=my_value, ...):
            ...
        the incoming OSC message should alternate argument names with values:
            /osc/route 'my_key' value 'my_key2' value ...
        
        Args:
            route: specify the OSC route. if None, use the function name
            json_keys: names of keyword arguments which should be decoded
                from JSON to python objects, 
                in the case that they arrive as strings.
                alternatively, if a string starts with '%JSON:' it will be decoded.
        """
        return self._decorate(True, route, return_host, return_port, json_keys)

    def __call__(self, client, *a, **kw):
        """alternate syntax for `send` with client name first"""
        self.send(*a, client=client, **kw)

class Updater():
    '''
    Rate-limited function call
    '''

    def __init__(self, f, count=30):
        self.f = f
        self.count = count
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.f()
            self.counter = 0

class ReceiveUpdater:
    '''
    Decouples event handling from updating
    Updating is rate-limited by a counter
    TODO: Handle return to sender
    '''

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state
        self.ret = None

    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True

    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and
                self.counter > self.count and
                self.state is not None):
            return
        self.ret = self.f(*self.state)
        '''
        if ret is not None:
            route = self.pascal_to_path(kwargs['name'])
            print('wrapper', route, ret, self.client_name)
            self.osc.return_to_sender_by_name((route, ret), self.client_name)
        '''
        self.counter = 0
        self.update = False
        return self.ret

class ReceiveListUpdater:
    '''
    Decouples event handling from updating
    Updating is rate-limited by a counter
    Assumes a list[float] instead of *args
    '''

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True

    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and
                self.counter > self.count and
                self.state is not None):
            return
        self.f(self.state)
        self.counter = 0
        self.update = False

class OSCReceiveUpdater(ReceiveUpdater):
    '''
    ReceiveUpdater with an OSC handler
    '''

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        # FIXME: ip:port/args
        '''
        v: first argument to the handler is the IP:port of the sender
        v: or you can use dispatcher.map directly
           and not set needs_reply_address=True
        j: can I get ip:port from osc itself?
        v: if you know the sender ahead of time yeah,
           but that lets you respond to different senders dynamically
        '''
        self.set(args[1:])

class OSCReceiveListUpdater(ReceiveListUpdater):
    '''
    ReceiveListUpdater with an OSC handler
    '''

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        self.set(list(args[1:]))


class OSCSend():
    '''
    Non rate-limited OSC send
    '''
    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.client = client

    def __call__(self, *args):
        self.osc.send(self.address, *self.f(*args), client=self.client)

class OSCSendUpdater():
    '''
    Rate-limited OSC send
    '''

    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.count = count
        self.counter = 0
        self.client = client

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.osc.send(self.address, *self.f(), client=self.client)
            self.counter = 0

class OSCReceiveUpdaters:
    '''
    o = OSCReceiveUpdaters(osc,
        {"/tolvera/particles/pos": s.osc_set_pos,
         "/tolvera/particles/vel": s.osc_set_vel})
    '''

    def __init__(self, osc, receives=None, count=10):
        self.osc = osc
        self.receives = []
        self.count = count
        if receives is not None:
            self.add_dict(receives, count=self.count)

    def add_dict(self, receives, count=None):
        if count is None:
            count = self.count
        {a: self.add(a, f, count=count) for a, f in receives.items()}

    def add(self, address, function, state=None, count=None, update=False):
        if count is None:
            count = self.count
        self.receives.append(
            OSCReceiveUpdater(self.osc, address, function,
                              state, count, update))

    def __call__(self):
        [r() for r in self.receives]


class OSCSendUpdaters:
    '''
    o = OSCSendUpdaters(osc, client="particles", count=10,
        sends={
            "/tolvera/particles/get/pos/all": s.osc_get_pos_all
        })
    '''

    def __init__(self, osc, sends=None, count=10, client=None):
        self.osc = osc
        self.sends = []
        self.count = count
        self.client = client
        if sends is not None:
            self.add_dict(sends, self.count, self.client)

    def add_dict(self, sends, count=None, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        {a: self.add(a, f, count=count, client=client)
                     for a, f in sends.items()}

    def add(self, address, function, state=None, count=None, update=False, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        self.sends.append(
            OSCSendUpdater(self.osc, address, function, count, client))

    def __call__(self):
        [s() for s in self.sends]


class OSCUpdaters:
    '''
    o = OSCUpdaters(osc, client="boids", count=10,
        receives={
            "/tolvera/boids/pos": b.osc_set_pos,
            "/tolvera/boids/vel": b.osc_set_vel
        },
        sends={
            "/tolvera/boids/pos/all": b.osc_get_all_pos
        }
    )
    '''

    def __init__(self, osc,
                 sends=None, receives=None,
                 send_count=60, receive_count=10,
                 client=None):
        self.osc = osc
        self.client = client
        self.send_count = send_count
        self.receive_count = receive_count
        self.sends = OSCSendUpdaters(
            self.osc, count=self.send_count, client=self.client)
        self.receives = OSCReceiveUpdaters(self.osc, count=self.receive_count)
        if sends is not None:
            self.add_sends(sends)
        if receives is not None:
            self.add_receives(receives)

    def add_sends(self, sends, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add_dict(sends, count, client)

    def add_send(self, send, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add(send, client=client, count=count)

    def add_receives(self, receives, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add_dict(receives, count=count)

    def add_receive(self, receive, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add(receive, count=count)

    def __call__(self):
        self.sends()
        self.receives()
