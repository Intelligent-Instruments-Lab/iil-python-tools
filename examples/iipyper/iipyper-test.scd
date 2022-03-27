(
MIDIIn.connectAll;
b = NetAddr.new("127.0.0.1", 9999);
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
// Server.default.options.inDevice_("mic-buds");
// Server.default.options.outDevice_("mic-buds");
// s.boot;
)

thisProcess.openUDPPort(57121);


(
// MIDI from controller
MIDIdef.noteOn(\input, {
    arg val, num, chan, src;
    t = Process.elapsedTime;
    [\input, num, val, chan, src].postln;
    b.sendMsg("/test", num, val);
    b.sendMsg("/keyword_example", \arg1, num, \arg2, val);
    b.sendMsg("/math/add", num, val);
    b.sendMsg("/math/mul", num, val);
});

// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    [\return, Process.elapsedTime-t, msg].postln;

}, "/test", nil);

OSCdef.newMatching(\add, {
    arg msg, time, addr, recvPort;
    [\add, Process.elapsedTime-t, msg].postln;
}, "/math/add", nil);

OSCdef.newMatching(\mul, {
    arg msg, time, addr, recvPort;
    [\mul, Process.elapsedTime-t, msg].postln;
}, "/math/mul", nil);

OSCdef(\default_send_test, {
    arg msg, time, addr, recvPort;
    [\default_send_test, msg].postln;
}, "/default_send_test", nil);

OSCdef(\other_send_test, {
    arg msg, time, addr, recvPort;
    [\other_send_test, msg].postln;
}, "/other_send_test", nil);
)

OSCFunc.trace(false)

// nil.postln