// in this example the model's predictions are fed back to it so it plays itself.
// the player can add notes as well and start/stop/reset the model with a footswitch.

// TODO: steerable generation. gui for ranges, temperatures;
// MIDI controller for pitch set

// TODO: keep track of pending model predictions so there are no races
// whenever a note is played, feed it to the model
// count the number of pending predictions
// if a note is played while there are pending predictions, cancel them
// ideally, would signal python to interrupt computation
// less complicated: just ignore the results when they come in
// there are pending predictions and pending notes:
// pending predictions need to be ignored when they come in
// pending notes need to be unscheduled

// TODO: pitch blacklist option
// use case: forbid model from releasing a note held by the player

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
    var freq = \freq.kr;
    var fl = freq.log2 - 1;
    var signal = Saw.ar(freq, 0.2) * EnvGate.new(1);
    var fr = 2.pow(Decay.ar(Impulse.ar(0), 3)*(13-fl)*vel+fl);
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
    \pitch, 60+12.rand, \time, 0, \vel, 0,
    \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1
);
)

// set the delay for more precise timing
~delay = 0.016;

// duet with the model
// feeds the model's predictions back to it as well as player input
(
~pending_predictions = 0;
~synths = nil!128;
~player_held = false!128;
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
        ~synths.do(_.release(1.0));
        SystemClock.clear;
        ~step = 0;
        };
    ~gate.postln;
});

// MIDI from controller
MIDIdef.noteOff(\input_off, {
    arg val, num, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-(t?(t2-~delay)); //time since last note

    // cancel any pending predictions
    SystemClock.clear;
    ~pending_predictions.postln;
    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict",
        \pitch, num, \time, dt, \vel, 0,
        \allow_start, false, \allow_end, false,
        \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1,
        \min_time, ~delay, \max_time, 5
        // \fix_time, ~delay
    );
    ~pending_predictions = ~pending_predictions+1;

    // release the previous note
    ~synths[num]!?(_.release(0.05));
    ~synths[num] = nil;

    // post the current note
    [\player, dt, num, 0].postln;

    // mark time of current note
    t = t2;
    ~player_t = t;

    ~step = ~step + 1;

    ~player_held[num] = false;

});
MIDIdef.noteOn(\input_on, {
    arg val, num, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-(t?(t2-~delay)); //time since last note

    // cancel any pending predictions
    SystemClock.clear;
    ~pending_predictions.postln;
    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict",
        \pitch, num, \time, dt, \vel, val,
        \allow_start, false, \allow_end, false,
        \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1,
        \min_time, ~delay, \max_time, 5
        // \fix_time, ~delay
    );
    ~pending_predictions = ~pending_predictions+1;


    // release the previous note
    ~synths[num]!?(_.release(0.05));

    // play the current note
    ~synths[num] = Synth(\pluck, [\freq, num.midicps, \vel, val/127]);//.release(1);

    // post the current note
    [\player, dt, num, val].postln;

    // mark time of current note
    t = t2;
    ~player_t = t;

    ~step = ~step + 1;
    // ~step = 0;

    ~player_held[num] = true;
});

// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    var num = msg[1]; // MIDI number of predicted note
    var dt = msg[2]; // time to predicted note
    var val = msg[3]; // velocity 0-127
    var step = msg[4];

    // time-to-next note gets 'censored' by the model
    // when over a threshold, in this case 10 seconds,
    // meaning it just predicts 10s rather than any longer time
    var censor = dt>=10.0;

    ~pending_predictions = ~pending_predictions-1;
    [\step, step].postln;


    censor.if{
        // if the predicted time is > 10 seconds, don't schedule it, just stop.
        \censor.postln;
        // ~synths[num]!?(_.release(3.0));
    }{
        // schedule the predicted note
        SystemClock.sched(dt-~delay, {
            (~gate>0).if{
                var t2 = Process.elapsedTime;
                var dt_actual = t2 - t;
                (num==129).if{
                    // 129 is the 'stop token',  meaning 'end-of-performance'
                    // in this case don't schedule a note, and reset the model
                    // b.sendMsg("/predictor/reset");
                    //release the last note
                    ~synths.do(_.release(1.0));
                    // unset time so next note will have dt=0
                    // t = nil;
                    // \reset.postln
                    \end.postln;
                }{
                    // cancel any pending predictions
                    // (there shouldn't be any, but might
                    // be if there was a lot of fast MIDI input)
                    SystemClock.clear;
                    ~pending_predictions.postln;
                    // feed model its own prediction as input
                    b.sendMsg("/predictor/predict",
                        \pitch, num, \time, dt_actual, \vel, val,
                        \allow_start, false, \allow_end, true,
                        \pitch_temp, 0.7, \rhythm_temp, 0.7, \timing_temp, 0.1,
                        \min_time, ~delay*2, \max_time, 5,
                        // \min_vel, 10
                        // \fix_time, ((~step+1)%3==0).if{0.6}{0}  // triads
                        // \fix_time, (~step%8)*0.1 // specific rhythm

                    );
                    ~pending_predictions = ~pending_predictions+1;

                    // play the current note
                    ~synths[num]!?(_.release(0.05));
                    (val > 0).if{
                        ~synths[num] = Synth(\pluck, [\freq, num.midicps, \vel, val/127])
                    }{
                        ~synths[num] = nil
                    };
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

(
// send a note manually if you don't have a MIDI controller:
SystemClock.clear;
~synths.do(_.release(1.0));
b.sendMsg("/predictor/reset");
{MIDIdef.all[\input].func.value(99, 60)}.defer(0.5);
SystemClock.clear;
)
// b.sendMsg("/predictor/predict", \pitch, 70, \time, 0, \vel, 64);

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");