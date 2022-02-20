from osc4py3.as_eventloop import *
from osc4py3 import oscmethod as osm

class OSCServer():
  """
  Provides a process callback for custom message handling
  """
  def __init__(self, ip=127.0.0.1, tx=8888, rx=9999, name="pytorch-osc", end_point="pytorch-osc", process=None, poll=False):
    # super().__init__():
    self.ip = ip
    self.tx = tx
    self.rx = rx
    self.name = name
    self.endpt = end_point
    self.connected = False
    self.process = process
    self.poll_timeout = 10 # seconds?
    # if poll is True: # enable polling when connected

  def _process(address, *args):
    """
    Process OSC messages from client
    Supplied as arg to osc_method()
    """
    if(address == self.endpt+"/handshake/request"):
      self.rx_handshake_request()
    else if(address == self.endpt+"/handshake/reply"):
      self.rx_handshake_reply()
    else if(address == self.endpt+"/poll"):
      self.poll()
    else if(self.process is not None):
      self.process(address, args)
    else:
      print('[OSCServer] No custom process() method supplied to handle message {} from address {}.'.format(args, address))

  def send(self, address, msg=None, print=False):
    """
    Send OSC message to client
    """
    if (self.connected is False):
      print('[OSCServer] Cannot send message: no client connected.')
    else:
      if (print is True):
        print('[OSCServer] Sending message {} to address {}.'.format(address, msg))
      osc_send(address, msg)

  def start(self):
    """
    Start OSCServer process
    TODO: Try / Event loop?
    TODO: 
    """
    print('[OSCServer] Started on {}:{}/{}'.format(self.ip, self.tx, self.endpt))
    osc_startup()
    osc_udp_server(self.ip, self.rx, self.name)
    osc_method(self.endpt, self._process, argscheme=osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK)

  def stop(self):
    """
    Stop OSCServer process
    """
    osc_terminate()
    print('[OSCServer] Stopped.')
    self.connected = False

  def rx_handshake_request(self):
    """
    Receive handshake request from client
    """
    self.connected = True
    print('[OSCServer] Received handshake request from client: client connected.')
    self.tx_handshake_reply()

  def rx_handshake_reply(self):
    """
    Receive handshake reply from client
    """
    self.connected = True
    print('[OSCServer] Received handshake reply from client: client connected.')

  def tx_handshake_request(self):
    """
    Request handshake from client
    """
    self.send(self.endpt+"/handshake/request")
    print('[OSCServer] Requesting handshake from client.')

  def tx_handshake_reply(self):
    """
    Reply to client handshake request
    """
    self.send(self.endpt+"/handshake/reply")
    print('[OSCServer] Replying to client handshake request.')

  def tx_poll(self):
    """
    TODO: poll connection, handle poll timer and timeout
    """
    self.send(self.endpt+"/poll")
    print('[OSCServer] Polling client connection...')

  def rx_poll(self):
    """
    TODO: reset poll timer
    """
    print('[OSCServer] Client still connected.')

  def poll_timed_out(self):
    """
    TODO
    """
    self.connected = False
    print('[OSCServer] Client polling timed out. Client assumed disconnected.')
