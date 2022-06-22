from typing import Tuple
import time
import json
import cProfile as profile
import inspect

from pythonosc import osc_packet
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

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

def convert_json(v, force, route):
    if not isinstance(v, str): return
    has_prefix = v.startswith('%JSON:')
    if has_prefix:
        v = v[6:]
    if force or has_prefix:
        try:
            return json.loads(v)
        except (TypeError, json.JSONDecodeError) as e:
            print(f"""
            warning: JSON decode failed for {route} argument "{k}": 
            value: {v}
            {type(e)} {e}
            """)
    return v

class OSC():
    """
    TODO: Handshake between server and clients
    TODO: Polling clients after handshake
    TODO: Enqueuing and buffering messages
    TODO: Allow multiple servers and clients
    """
    instances = []
    def __init__(self, host="127.0.0.1", port=9999, verbose=True):
        """
        TODO: Expand to support multiple IPs + ports

        Args:
            host (str): IP address
            port (int): port to receive on
            verbose (bool): whether to print activity
        """
        self.verbose = verbose
        self.host = host
        self.port = port
        self.dispatcher = Dispatcher()
        self.server = None
        self.clients = {} # (host,port) -> client
        self.client_names = {} # (name) -> (host,port)

        # self._pending_handlers = []

        OSC.instances.append(self)

    async def create_server(self, event_loop):#, host=None, port=None):
        """
        Create the server
        """
        # if (host is None):
        #     host = self.host
        # if (port is None):
        #     port = self.port
        if (self.server is None):
            self.server = AsyncIOOSCUDPServer(
                (self.host, self.port), self.dispatcher, event_loop)
            self.transport, self.protocol = await self.server.create_serve_endpoint()
            if self.verbose:
                print(f"OSC server created {self.host}:{self.port}")
            # for item in self._pending_handlers:
                # self.add_handler(*item)
        else:
            print("OSC server already exists")

    def close_server(self):
        """
        Close the server
        """
        if (self.server is not None):
            self.transport.close()
        else:
            print("OSC server does not exist")

    def add_handler(self, address, handler):
        """
        Map the custom message handler to the OSC dispatcher
        """
        # if (self.server is not None):
        self.dispatcher.map(address, handler, needs_reply_address=True)
        # else:
            # self._pending_handlers.append((address, handler))
            # print("OSC server does not exist")

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
            address: '/osc/route' or 'host:port/osc/route'
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

    def _decorate(self, arg_mode, route, pass_route,
            return_host, return_port, return_route, json_keys):
        """generic decorator (args and kwargs cases)"""
        if hasattr(route, '__call__'):
            # bare decorator
            f = route
            route = None
            json_keys = set()
        else:
            f = None
            json_keys = set(json_keys or [])

        def decorator(f, route=route, pass_route=pass_route,
                return_host=return_host, return_port=return_port, 
                return_route=return_route,
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
                if arg_mode=='args':
                    kwargs = {k:v for k,v in zip(args[::2], args[1::2])}
                    # JSON conversions
                    for k in kwargs: 
                        kwargs[k] = convert_json(
                            kwargs[k], k in json_keys, route)
                    args = []
                elif arg_mode=='kwargs':
                    # e.g. "arg1", 0, "arg2", 1
                    # -> kwargs = {"arg1":0 "arg2":1}
                    args = [convert_json(a, False, route) for a in args]
                    kwargs = {}
                elif arg_mode=='vectors':
                    # e.g. "arg1", 0, 1, "arg2", 0.5
                    # -> kwargs = {"arg1":[0,1], "arg2":[0.5]}
                    s = inspect.signature(f)
                    kwargs = {}
                    v = None
                    for item in args:
                        # if item is a keyword name,
                        # start collecting into a list
                        if item in s.parameters:
                            v = []
                            kwargs[item] = v
                        else:
                            # this will allow strings and mized types in
                            # arguments, as long as they don't collide
                            # with argument names in the function
                            if v is None:
                                raise ValueError(f"""
                                unexpected argument "{item}" to route "{route}"
                                """)
                            v.append(item)
                    args = []
                    # print(args,kwargs)

                if pass_route:
                    r = f(address, *args, **kwargs)
                else:
                    # t = time.time()
                    r = f(*args, **kwargs)
                    # print(time.time() - t)
                    # profile.runctx('r = f(*args, **kwargs)', globals(), locals())

                # if there was a return value,
                # send it as a message back to the sender
                if r is None:
                    return
                # if return route not given in decorator,
                # use first returned value
                if return_route is None:
                    if (
                        not hasattr(r, '__len__') or 
                        not isinstance(r[0], str) or
                        len(r)<2):
                        print("""
                        value returned from OSC handler should start with route
                        """)
                    rr, *r = r
                else:
                    rr = return_route
                client = (
                    client[0] if return_host is None else return_host,
                    client[1] if return_port is None else return_port
                )
                print(client, r)
                self.get_client_by_sender(client).send_message(rr, r)

            self.add_handler(route, handler)

            return f

        return decorator if f is None else decorator(f)
    
    def args(self, route=None, pass_route=True,
        return_host=None, return_port=None, return_route=None):
        """decorate a function as an args-style OSC handler.

        each item of the OSC message is passed as a positional argument.
        if the function returns a value, it will be sent as a reply to the
        OSC sender.

        Args:
            route: if None, route will be derived from function name
            pass_route: if True, route will be passed as first argument
            return_host: override host address for return message
            return_port: override port for return message
            return_route: if None, return route is expected to be first
                returned value

        """
        return self._decorate('args', route, pass_route,
            return_host, return_port, return_route, None)

    def kwargs(self, route=None, pass_route=True,
        return_host=None, return_port=None, return_route=None, 
        json_keys=None):
        """decorate a function as an kwargs-style OSC handler
        
        Args:
            route: specify the OSC route. if None, use the function name
            json_keys: names of keyword arguments which should be decoded
                from JSON, in the case that they arrive as strings
        """
        return self._decorate('kwargs', route, pass_route,
            return_host, return_port, return_route, json_keys)

    def vectors(self, route=None, pass_route=True,
        return_host=None, return_port=None, return_route=None):
        """decorate a function as an args-style OSC handler.

        each item of the OSC message is passed as a positional argument.
        if the function returns a value, it will be sent as a reply to the
        OSC sender.

        Args:
            route: if None, route will be derived from function name
            pass_route: if True, route will be passed as first argument
            return_host: override host address for return message
            return_port: override port for return message
            return_route: if None, return route is expected to be first
                returned value

        """
        return self._decorate('vectors', route, pass_route,
            return_host, return_port, return_route, None)

    def api(self, obj, route=None, arg_mode='kwargs', pass_route=False):
        """wrap an instance of a class, adding an OSC route for each of its methods"""
        method_names = [m for m in dir(obj) if not m.startswith('_')]
        if route is None:
            route = type(obj).__name__
        for m in method_names:
            method=getattr(obj, m)
            if callable(method):
                m_route = '/'.join(('', route, m))
                print(m_route)
                if arg_mode=='kwargs':
                    self.kwargs(
                        route=m_route, return_route=m_route, 
                        pass_route=pass_route)(method)
                elif arg_mode=='args':
                    self.args(
                        route=m_route, return_route=m_route,
                        pass_route=pass_route)(method)
                elif arg_mode=='vectors':
                    self.vectors(
                        route=m_route, return_route=m_route,
                        pass_route=pass_route)(method)

    def __call__(self, *a, **kw):
        """syntactic sugar:
        
        alternate syntax for `send` with client name first;
        
        or alias for `cls`
        """
        if isinstance(a[0], str):
            client, *a = a
            self.send(*a, client=client, **kw)
        else:
            self.api(*a, **kw)
