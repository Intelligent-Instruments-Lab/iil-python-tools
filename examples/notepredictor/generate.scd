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

MIDIClient.init
MIDIClient.destinations
~m1 = MIDIOut.newByName("IAC Driver", "Bus 1");
~m2 = MIDIOut.newByName("IAC Driver", "IAC Bus 2");
/*
~m1.noteOn(0, 60, 127)
~m2.noteOn(0, 60, 127)
~m1.noteOff(0, 60, 64)
~m2.noteOff(0, 60, 64)*/

(
~prog2portchan = { arg prog;
    case
    {prog==0}{"can't convert start token".postln}
    {prog<=128}{
        var group = (prog-1 /8).asInteger;
        var idx = (prog-1 %8).asInteger;
        var port = switch(group)
        { 0}{(idx<6).if{~m1}{~m2}} //piano
        { 1}{((idx<3)||(idx==5)).if{~m1}{~m2}} //chromatic perc
        { 2}{(idx<4).if{~m1}{~m2}} //organ
        { 3}{(idx<5).if{~m1}{~m2}} //guitar
        { 4}{(idx<4).if{~m2}{~m1}} //bass
        { 5}{(idx<5).if{~m1}{~m2}} //strings
        { 6}{(idx<4).if{~m1}{~m2}} //ensemble
        { 7}{(idx<3).if{~m1}{~m2}} //brass
        { 8}{(idx<4).if{~m1}{~m2}} //reed
        { 9}{(idx<3).if{~m1}{~m2}} //pipe
        {10}{(idx%2==0).if{~m1}{~m2}} //synth lead
        {11}{(idx<4).if{~m1}{~m2}} //synth pad
        {12}{~m1} //synth fx
        {13}{case //'ethnic'
            {idx<=3}{~m1}
            {idx==4}{~m2}
            {idx==5}{group=2;~m2}
            {idx==6}{group=5;~m1}
            {idx==7}{group=8;~m2}
        }
        {14}{~m1} //percussive
        {15}{~m1} //sound fx
        ;
        // \melody.postln;
        (port:port, chan:group)
    }{(prog<=256)||(prog>=265)}{
        // \drum.postln;
        (port:~m2, chan:12)
    }{prog-257 < 8}{
        // \anon.postln;
        (port:~m1, chan:0)
    }
};
~release_all = {arg vel=0;
    [~m1, ~m2].do{arg port; 128.do{arg note; 16.do{arg chan; port.noteOff(chan, note, vel)}}}
};
)

/*(
~prog2portchan.(257)[\chan]
)*/

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
    \inst, 0, \pitch, 60+12.rand, \time, 0, \vel, 0,
    \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1
);
)

// set the delay for more precise timing
~delay = 0.02;

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
    var inst = ~player_inst;
    var port_chan = ~prog2portchan.(inst);

    // cancel any pending predictions
    SystemClock.clear;
    ~pending_predictions.postln;
    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict",
        \inst, inst, \pitch, num, \time, dt, \vel, 0,
        \allow_start, false, \allow_end, false,
        \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1,
        \min_time, ~delay, \max_time, 5
        // \fix_time, ~delay
    );
    ~pending_predictions = ~pending_predictions+1;

    // release the previous note
    ~synths[num]!?(_.release(0.05));
    ~synths[num] = nil;
    // send MIDI
    port_chan[\port].noteOff(port_chan[\chan], num);

    // post the current note
    [\player, dt, inst, num, 0].postln;

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
    var inst = ~player_inst;
    var port_chan = ~prog2portchan.(inst);


    // cancel any pending predictions
    SystemClock.clear;
    ~pending_predictions.postln;
    //get a new prediction in light of current note
    b.sendMsg("/predictor/predict",
        \inst, inst, \pitch, num, \time, dt, \vel, val,
        \allow_start, false, \allow_end, false,
        \pitch_temp, 0.5, \rhythm_temp, 0.5, \timing_temp, 0.1,
        \min_time, ~delay, \max_time, 5
        // \fix_time, ~delay
    );
    ~pending_predictions = ~pending_predictions+1;


    // release the previous note (if not properly released by noteoff)
    ~synths[num]!?(_.release(0.05));

    // play the current note
    ~synths[num] = Synth(\pluck, [\freq, num.midicps, \vel, val/127]);//.release(1);
    // send MIDI
    port_chan[\port].noteOn(port_chan[\chan], num, val);

    // post the current note
    [\player, dt, inst, num, val].postln;

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
    var inst = msg[1]; // instrument of predicted note
    var pitch = msg[2]; // MIDI number of predicted note
    var dt = msg[3]; // time to predicted note
    var vel = msg[4]; // velocity 0-127
    var end = msg[5];
    var step = msg[6];
    var port_chan = ~prog2portchan.(inst);


    // time-to-next note gets 'censored' by the model
    // when over a threshold, in this case 10 seconds,
    // meaning it just predicts 10s rather than any longer time
    var censor = dt>=10.0;

    ~pending_predictions = ~pending_predictions-1;
    // [\step, step].postln;


    censor.if{
        // if the predicted time is > 10 seconds, don't schedule it, just stop.
        \censor.postln;
        // ~synths[pitch]!?(_.release(3.0));
    }{
        // schedule the predicted note
        SystemClock.sched(dt-~delay, {
            (~gate>0).if{
                var t2 = Process.elapsedTime;
                var dt_actual = t2 - t;
                (end==1).if{
                    // 129 is the 'stop token',  meaning 'end-of-performance'
                    // in this case don't schedule a note, and reset the model
                    // b.sendMsg("/predictor/reset");
                    //release the last note
                    ~synths.do(_.release(1.0));
                    ~release_all.(127);
                    // unset time so next note will have dt=0
                    // t = nil;
                    // \reset.postln
                    \end.postln;
                }{
                    // cancel any pending predictions
                    // (there shouldn't be any, but might
                    // be if there was a lot of fast MIDI input)
                    SystemClock.clear;
                    // ~pending_predictions.postln;
                    // feed model its own prediction as input
                    b.sendMsg("/predictor/predict",
                        \inst, inst, \pitch, pitch, \time, dt_actual, \vel, vel,
                        \allow_start, false, \allow_end, true,
                        \instrument_temp, 1, \pitch_temp, 0.9, \rhythm_temp, 0.7, \timing_temp, 0.05,
                        // \instrument_temp, 1, \pitch_temp, 1, \rhythm_temp, 1, \timing_temp, 1,
                        \min_time, ~delay,
                        \max_time, 5,

                    );
                    ~pending_predictions = ~pending_predictions+1;

                    // play the current note
                    ~synths[pitch]!?(_.release(0.05));
                    // send MIDI
                    port_chan[\port].noteOff(port_chan[\chan], pitch, 64);
                    (vel > 0).if{
                        ~synths[pitch] = Synth(\pluck, [
                            \freq, pitch.midicps, \vel, vel/127]);
                        // send MIDI
                        port_chan[\port].noteOn(port_chan[\chan], pitch, vel)
                    }{
                        ~synths[pitch] = nil;
                    };
                    // post the current note
                    [\model, step, dt, inst, pitch, vel, end].postln;
                    // mark the actual time of current note
                    t = t2;
                    ~machine_t = t;
                    // crudely draw note on piano GUI
                    ~gui.if{
                        AppClock.sched(0,{k.keyDown(pitch)});
                        AppClock.sched(0.2,{k.keyUp(pitch)});
                    }
                };
                ~step = ~step+1;
                // [\late, dt_actual-dt].postln;
            }
    })};

}, "/prediction", nil);
)

(
// send a note manually if you don't have a MIDI controller:
SystemClock.clear;
~synths.do(_.release(1.0));
~release_all.(0);
b.sendMsg("/predictor/reset");
// ~player_inst = 272;
// {MIDIdef.all[\input_on].func.value(99, 83)}.defer(0.5);
~player_inst = 128.rand+1;
{MIDIdef.all[\input_on].func.value(128.rand, 128.rand)}.defer(0.1);
SystemClock.clear;
)
// b.sendMsg("/predictor/predict", \pitch, 70, \time, 0, \vel, 64);

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");