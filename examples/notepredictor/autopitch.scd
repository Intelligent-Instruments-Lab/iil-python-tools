(
~gui = false;
MIDIIn.connectAll;
b = NetAddr.new("127.0.0.1", 9999);
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
// Server.default.options.inDevice_("mic-buds");
// Server.default.options.outDevice_("mic-buds");
s.boot;
~gui.if{
    k = MIDIKeyboard.new(bounds: Rect(0, 0, 500, 100), octaves:11, startnote:0)
};
)

(
SynthDef(\pluck, {
    var vel = \vel.kr;
    var signal = Saw.ar(\freq.kr(20), 3e-2) * EnvGate.new(1);
    var fr = 2.pow(Decay.ar(Impulse.ar(0), 3)*6*vel+8);
    signal = BLowPass.ar(signal, fr)*vel;
    Out.ar([0,1], signal);
}).add
)


// measure round-trip latency
(
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    (Process.elapsedTime - t).postln;
}, '/prediction', nil);
t = Process.elapsedTime;
b.sendMsg("/predictor/predict",
    \pitch, 60+12.rand, \time, 0, \vel, 0, \fix_time, 0, \fix_vel, 0);
)

// set the delay for more precise timing
~delay = 0.01;


// NetAddr.localAddr    // retrieve the current IP and port
// thisProcess.openPorts; // list all open ports

// model chooses pitches
(
~gate = 1;

~reset = {
    ~last_pitch = nil;
    ~last_dt = nil;
    ~last_vel = nil;
    t = Process.elapsedTime;
    b.sendMsg("/predictor/reset");
    y!?{y.free};
    y = nil;
    b.sendMsg("/predictor/predict", \pitch, 128, \time, 0, \vel, 0);

};

~reset.();

// footswitch
MIDIdef.program(\switch, {
    arg num, chan, src;
    num.switch
    {1}{~gate = 0}
    {2}{~gate = 1}
    {3}{
        ~gate = 0;
        SystemClock.clear;
        b.sendMsg("/predictor/reset");
        y.release;
        SystemClock.clear;
        };
    ~gate.postln;
});


// MIDI from controller
MIDIdef.noteOn(\input, {
    arg vel, pitch, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-(t?t2); //time since last note

    // release the previous note
    y.release(0.1);

    // attack the current note with the old pitch
    y = Synth(\pluck, [\freq, ~last_pitch.midicps, \vel, vel/127]);

    // get a new prediction in light of last note,
    // fixing dt and vel to performed values so just pitch is predicted
    b.sendMsg("/predictor/predict",
        \pitch, ~last_pitch, \time, ~last_dt, \vel, ~last_vel,
        \fix_time, dt, \fix_vel, vel);

    ~last_dt = dt;
    ~last_vel = vel;

    // mark time of current note
    t = t2;
});

// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    var pitch = msg[1]; // MIDI number of predicted note
    var dt = msg[2]; // time to predicted note
    var vel = msg[3]; // velocity 0-127

    // store the pitch and immediately set (unless there is no synth,
    // indicating this is the first note)
    ~last_pitch = pitch;
    ~last_dt.isNil.if{~last_dt = dt};
    ~last_vel.isNil.if{~last_vel = vel};
    y!?{y.set(\freq, ~last_pitch.midicps)};

    [pitch, dt, vel].postln;

}, "/prediction", nil);
)

~reset.()
// send a note manually if you don't have a midi controller
MIDIdef.all[\input].func.(64)

(
MIDIdef.all[\input].func.(64);
MIDIdef.all[\input].func.(64);
)

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");