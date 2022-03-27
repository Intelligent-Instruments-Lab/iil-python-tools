from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

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

    def _decorate(self, use_kwargs, route):
        """generic decorator (args and kwargs cases)"""
        if hasattr(route, '__call__'):
            # bare decorator
            f = route
            route = None
        else:
            f = None

        def decorator(f, route=route):
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
                print('handler:', client, address)
                if use_kwargs:
                    kwargs = {k:v for k,v in zip(args[::2], args[1::2])}
                    args = []
                else:
                    kwargs = {}
                r = f(address, *args, **kwargs)
                # if there was a return value,
                # send it as a message back to the sender
                if r is not None:
                    if not hasattr(r, '__len__'):
                        print("""
                        value returned from OSC handler should start with route
                        """)
                    else:
                        self.get_client_by_sender(client).send_message(r[0], r[1:])

            self.add_handler(route, handler)

            return f

        return decorator if f is None else decorator(f)
    
    def args(self, route=None):
        """decorate a function as an args-style OSC handler."""
        return self._decorate(False, route)

    def kwargs(self, route=None):
        """decorate a function as an kwargs-style OSC handler"""
        return self._decorate(True, route)

    def __call__(self, client, *a, **kw):
        """alternate syntax for `send` with client name first"""
        self.send(*a, client=client, **kw)
