b = NetAddr.new("127.0.0.1", 9999);

b.sendMsg("/predictor/reset");

b.sendMsg("/predictor/forward", [1,2,3]);

b.sendMsg("/predictor/predict", 1);


// To listen to another application, that application needs to send a message to the port SuperCollider is listening on. Normally the default port is 57120, but it could be something different if that port was already bound when SC started. The current default port can be retrieved with

NetAddr.langPort;    // retrieve the current port SC is listening to

// Or you can retrieve both the IP and the port with:

NetAddr.localAddr    // retrieve the current IP and port

// You can open additional ports using Main: -openUDPPort. This will return a Boolean indicating whether SC succeeded in opening the new port. Or you can just pass a custom port as the recvPort argument to OSCFunc and it will open it automatically if not already open.

thisProcess.openUDPPort(1121); // attempt to open 1121

thisProcess.openPorts; // list all open ports

// To listen to incoming messages, an OSCFunc needs to be created in SuperCollider. If the sending application has a fixed port it sends message from, you can set the OSCFunc to listen only to messages coming from that IP and port:

n = NetAddr.new("127.0.0.1", 7771);    // create the NetAddr

// create the OSCFunc

o = OSCFunc({ arg msg, time, addr, recvPort; [msg, time, addr, recvPort].postln; }, '/goodbye', n);

o.free;    // remove the OSCFunc when you are done.
