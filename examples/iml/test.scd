b = NetAddr("localhost", 9999)


b.sendMsg("/IML/add", \source, 0, 1, \target, 3.14)
b.sendMsg("/IML/add", \source, 1, 1, \target, -1)

(
t = SystemClock.seconds;
b.sendMsg("/IML/map", \source, 0.5, 1)
)


(
OSCdef(\ret, {
    arg msg, time, addr, port;
    var result = msg[1..];
    (SystemClock.seconds - t).postln;
    [msg, time, addr, port].postln
}, "/IML/map")
)

// b.sendMsg("/IML/add", "%JSON:[0,1]", "?%JSON:[3.14]")
// b.sendMsg("/IML/add", "%JSON:[1,1]", "%JSON:[-1]")

// b.sendMsg("/IML/map", "%JSON:[0.5,1]")