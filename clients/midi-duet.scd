(
MIDIIn.connectAll;
b = NetAddr.new("127.0.0.1", 9999);
// Server.default.options.inDevice_("Built-in Microph");
// Server.default.options.outDevice_("Built-in Output");
Server.default.options.inDevice_("mic-buds");
Server.default.options.outDevice_("mic-buds");
s.boot;
k = MIDIKeyboard.new(bounds: Rect(0, 0, 500, 100), octaves:11, startnote:0)
)

(
SynthDef(\pluck, {
    var signal = Saw.ar(\freq.kr, 0.2) * EnvGate.new(1);
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
b.sendMsg("/predictor/predict", 60+12.rand, 0);
)

// set the delay for more precise timing
~delay = 0.015;

// NetAddr.localAddr    // retrieve the current IP and port
// thisProcess.openPorts; // list all open ports

// duet with the model
// feeds the model's predictions back to it as well as player input
(
t = Process.elapsedTime;
b.sendMsg("/predictor/reset");

// MIDI from controller
MIDIdef.noteOn(\input, {
    arg val, num, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-t; //time since last note

    // cancel any pending predictions
    SystemClock.clear;

    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict", num, dt);

    // release the previous note
    y.release(1.0);

    // play the current note
    y = Synth(\pluck, [\freq, num.midicps]);//.release(1);

    // post the current note
    [\player, dt, num].postln;

    // mark time of current note
    t = t2;
});


// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    var num = msg[1]; // MIDI number of predicted note
    var dt = msg[2]; // time to predicted note

    // time-to-next note gets 'censored' by the model
    // when over a threshold, in this case 10 seconds,
    // meaning it just predicts 10s rather than a any longer time
    var censor = dt==10.0;

    censor.if{
        // if the predicted time is > 10 seconds, don't schedule it, just stop.
        \censor.postln; y.release(3.0)
    }{
        // schedule the predicted note
        SystemClock.sched(dt-~delay, {
            (num==129).if{
                // 129 is the 'stop token',  meaning 'end-of-performance'
                // in this case don't schedule a note, and reset the model
                b.sendMsg("/predictor/reset");
                //release the last note
                y.release(1.0);
                \reset.postln
            }{
                // cancel any pending predictions (there shouldn't be any, but might
                // be if there was a lot of fast MIDI input)
                SystemClock.clear;
                // feed model its own prediction as input
                b.sendMsg("/predictor/predict", num, dt);
                // release the previous note
                (dt<1e-2).if{
                    // if the time delay is very small, slowly release to play a chord
                    y.release(1.0)
                }{
                    // otherwise release fast to play a melody
                    y.release(0.1)
                };
                // play the current note
                y = Synth(\pluck, [\freq, num.midicps]);//.release(1);
                // post the current note
                [\model, dt, num].postln;
                // mark the time of current note
                t = Process.elapsedTime;
                // crudely draw note on piano GUI
                AppClock.sched(0,{k.keyDown(num)});
                AppClock.sched(0.2,{k.keyUp(num)});
            }
    })};

}, '/prediction', nil);
)

// b.sendMsg("/predictor/predict", 70, 0.1);

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");