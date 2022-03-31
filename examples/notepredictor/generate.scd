// in this example the model's predictions are fed back to it so it plays itself.
// the player can add notes as well and start/stop/reset the model with a footswitch.

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
    var signal = Saw.ar(\freq.kr, 0.2) * EnvGate.new(1);
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
b.sendMsg("/predictor/predict", \pitch, 60+12.rand, \time, 0, \vel, 0);
)

// set the delay for more precise timing
~delay = 0.015;

// duet with the model
// feeds the model's predictions back to it as well as player input
(
~step = 0;
~gate = 1;
t = nil;
~player_t = t;
~machine_t = t;
b.sendMsg("/predictor/reset");

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
        ~step = 0;
        };
    ~gate.postln;
});


// MIDI from controller
MIDIdef.noteOn(\input, {
    arg val, num, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-(t?t2); //time since last note

    // cancel any pending predictions
    SystemClock.clear;

    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict",
        \pitch, num, \time, dt, \vel, val,
        \allow_start, false, \allow_end, false,
        \time_temp, 0, \min_time, 0.1, \max_time, 5
        // \fix_time, 9
    );

    // release the previous note
    y.release(0.1);

    // play the current note
    y = Synth(\pluck, [\freq, num.midicps, \vel, val/127]);//.release(1);

    // post the current note
    [\player, dt, num, val].postln;

    // mark time of current note
    t = t2;
    ~player_t = t;

    ~step = ~step + 1;
});


// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    var num = msg[1]; // MIDI number of predicted note
    var dt = msg[2]; // time to predicted note
    var val = msg[3]; // velocity 0-127

    // time-to-next note gets 'censored' by the model
    // when over a threshold, in this case 10 seconds,
    // meaning it just predicts 10s rather than any longer time
    var censor = dt>10.0;

    censor.if{
        // if the predicted time is > 10 seconds, don't schedule it, just stop.
        \censor.postln; y.release(3.0)
    }{
        // schedule the predicted note
        SystemClock.sched(dt-~delay, {
            (~gate>0).if{
                var t2 = Process.elapsedTime;
                var dt_actual = t2 - t;
                (num==129).if{
                    // 129 is the 'stop token',  meaning 'end-of-performance'
                    // in this case don't schedule a note, and reset the model
                    b.sendMsg("/predictor/reset");
                    //release the last note
                    y.release(1.0);
                    // unset time so next note will have dt=0
                    t = nil;
                    \reset.postln
                }{
                    // cancel any pending predictions
                    // (there shouldn't be any, but might
                    // be if there was a lot of fast MIDI input)
                    SystemClock.clear;
                    // feed model its own prediction as input
                    b.sendMsg("/predictor/predict",
                        \pitch, num, \time, dt_actual, \vel, val,
                        \allow_start, false, \allow_end, false,
                        \time_temp, 0.1, \min_time, 0.1, \max_time, 5
                        // \fix_time, (~step%4==0).if{0.6}{0}  // tetrachords
                        // \fix_time, (~step%8)*0.1 // specific rhythm

                    );
                    // release the previous note
                    (dt<3e-2).if{
                        // if the time delay is very small, slow release for chord
                        y.release(1.0)
                    }{
                        // otherwise release fast to play a melody
                        y.release(0.1)
                    };
                    // play the current note
                    y = Synth(\pluck, [
                        \freq, num.midicps, \vel, val/127]);//.release(1);
                    // post the current note
                    [\model, dt, num, val].postln;
                    // mark the actual time of current note
                    t = t2;
                    ~machine_t = t;
                    // crudely draw note on piano GUI
                    ~gui.if{
                        AppClock.sched(0,{k.keyDown(num)});
                        AppClock.sched(0.2,{k.keyUp(num)});
                    }
                };
                ~step = ~step+1;
                [\late, dt_actual-dt].postln;
            }
    })};

}, "/prediction", nil);
)

// send a note manually if you don't have a MIDI controller:
// b.sendMsg("/predictor/predict", \pitch, 70, \time, 0, \vel, 64);

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");