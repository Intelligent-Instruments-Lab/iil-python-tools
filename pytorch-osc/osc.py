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
    def __init__(self, ip="127.0.0.1", send=8888, receive=9999):
        """
        TODO: Expand to support multiple IPs + ports
        """
        self.ip = ip
        self.send = send
        self.receive = receive
        self.dispatcher = Dispatcher()
        self.server = None
        self.client = None
    
    async def createServer(self, event_loop, ip=None, port=None):
        """
        Create the server
        """
        if (ip == None):
            ip = self.ip
        if (port == None):
            port = self.receive
        if (self.server == None):
            self.server = AsyncIOOSCUDPServer((ip, port), self.dispatcher, event_loop)
            self.transport, self.protocol = await self.server.create_serve_endpoint()
            print(f"OSC serrver created {ip}:{port}")
        else:
            print("OSC server already exists")

    def closeServer(self):
        """
        Close the server
        """
        if (self.server != None):
            self.transport.close()
        else:
            print("OSC server does not exist")

    def addHandler(self, address, handler):
        """
        Map the custom message handler to the OSC dispatcher
        """
        if (self.server != None):
            self.dispatcher.map(address, handler)
        else:
            print("OSC server does not exist")

    def createClient(self, ip=None, port=None):
        """
        Add an OSC client
        """
        if (ip == None):
            ip = self.ip
        if (port == None):
            port = self.send
        if (self.client == None):
            self.client = SimpleUDPClient(ip, port)
            print(f"OSC client created {ip}:{port}")
        else:
            print("OSC client already exists")

    def sendMessage(self, address, msg):
        """
        Send message to client
        """
        if (self.client != None):
            self.client.send_message(address, msg)
            print(f"OSC message sent {address}:{msg}")
        else:
            print("OSC client does not exist")
