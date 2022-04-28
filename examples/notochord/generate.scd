// in this example the model's predictions are fed back to it so it plays itself.
// the player can add notes as well and start/stop/reset the model with a footswitch.

// TODO: steerable generation. gui for ranges, temperatures;
// MIDI controller for pitch set

(
MIDIClient.init;
// MIDIClient.destinations
~m1 = MIDIOut.newByName("IAC Driver", "Bus 1").latency_(0);
~m2 = MIDIOut.newByName("IAC Driver", "Bus 2").latency_(0);
~m3 = MIDIOut.newByName("IAC Driver", "Bus 3").latency_(0);
~gui = false;
MIDIIn.connectAll;
b = NetAddr.new("127.0.0.1", 9999);
~controller = MIDIClient.sources.detect { |e| e.device.containsi("MPKmini2") }.uid;
)


(
// TODO: what is with this input latency
MIDIdef.noteOn(\input_on, {
    arg vel, pitch, chan, src;
    var perftime = Process.elapsedTime;
    ~m2.noteOn(12, pitch, 100);
    (Process.elapsedTime - perftime).postln;
}, srcID:~controller);
)
(
// mapp instruments to channels + buses for the Ableton set
~prog2portchan = { arg prog;
    case
    {prog==0}{"can't convert start token".postln}
    {prog<=128}{
        var group = (prog-1 /8).asInteger;
        var idx = (prog-1 %8).asInteger;
        var port = switch(group)
        { 0}{case
            {idx<4}{~m1} //acoustic
            {idx<6}{~m2} //electric
            {true}{~m3} //harpsichord
        } //piano
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
    [~m1, ~m2, ~m3].do{arg port; 128.do{arg note; 16.do{arg chan;
        port.noteOff(chan, note, vel)}}}
};
)

// ~m1.noteOn(12, 60, 64);

/*(
~prog2portchan.(257)[\chan]
)*/

(
// Server.default.options.inDevice_("Built-in Microph");
// Server.default.options.outDevice_("Built-in Output");
// Server.default.options.inDevice_("mic-buds");
// Server.default.options.outDevice_("mic-buds");
// s.boot;
/*~gui.if{
    k = MIDIKeyboard.new(bounds: Rect(0, 0, 500, 100), octaves:11, startnote:0)
};*/
)

/*(
SynthDef(\pluck, {
    var vel = \vel.kr;
    var freq = \freq.kr;
    var fl = freq.log2 - 1;
    var signal = Saw.ar(freq, 0.2) * EnvGate.new(1);
    var fr = 2.pow(Decay.ar(Impulse.ar(0), 3)*(13-fl)*vel+fl);
    signal = BLowPass.ar(signal, fr)*vel;
    Out.ar([0,1], signal);
}).add
)*/

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
~delay = 0.01;

// duet with the model
// feeds the model's predictions back to it as well as player input
(
~velocity_temp = 1;
~instrument_temp = 0.95;
~pitch_temp = 0.9;
~rhythm_temp = 1;
~timing_temp = 0.05;
~min_time = nil;
~max_time = 5;
~allow_end = true;
~player_sound = true;
~player_instrument = 1;
~use_synth = false;
~duet = false;
~include_pitch = nil;

~gate = true;

~synths = nil!128;
~player_held = false!128;

~pending_predictions = 0;
~pending_events = List[];
~step = 0;

t = nil;
~player_t = t;
~machine_t = t;

b.sendMsg("/predictor/reset");

~global_kws = {
    //add a collection of global variables to an argument list
    arg kw, names;
    names.do{arg name;
        var val = currentEnvironment[name];
        val.notNil.if{
            kw.add(name);
            // convert Collections to JSON strings
            val.isKindOf(Collection).if{
                val = "%JSON:"++JSON.stringify(val)};
            kw.add(val)}
    }
};


~predict_event = {
    arg inst, pitch, dt, vel;
    var kw = List[
        \inst, inst, \pitch, pitch, \time, dt, \vel, vel,
        // \allow_start, false
    ];
    ~duet.if{kw.add(\exclude_instrument); kw.add(~player_inst)};
    ~global_kws.(kw, [
        \min_time, \max_time, \allow_end,
        \instrument_temp, \pitch_temp, \rhythm_temp, \timing_temp,
        \velocity_temp, \include_pitch
    ]);

    // kw.postln;

    b.sendMsg("/predictor/predict", *kw);
    ~pending_predictions = ~pending_predictions+1;

};

~clear_pending_events = {~pending_events.do{arg pe; pe.stop}.clear};

// footswitch
/*MIDIdef.program(\switch, {
    arg num, chan, src;
    num.switch
    {1}{~gate = false}
    {2}{~gate = true}
    {3}{
        ~gate = false;
        SystemClock.clear;
        b.sendMsg("/predictor/reset");
        ~synths.do(_.release(1.0));
        SystemClock.clear;
        ~step = 0;
        };
    ~gate.postln;
});*/

// MIDI from controller
MIDIdef.noteOff(\input_off, {
    arg vel, pitch, chan, src;
    var t2, dt;
    var inst = ~player_inst;
    var port_chan = ~prog2portchan.(inst);

    // mark time of current note
    t2 = Process.elapsedTime;
    dt = t2-(t?(t2-~delay));
    t = t2;
    ~player_t = t;

    // cancel any pending predictions
    ~clear_pending_events.();
    // ~pending_predictions.postln;

    ~player_sound.if{
        // release the previous note
        ~synths[pitch]!?(_.release(0.05));
        ~synths[pitch] = nil;
        // send MIDI
        port_chan[\port].noteOff(port_chan[\chan], pitch);
    };

    //get a new prediction in light of current note
    ~predict_event.(inst, pitch, dt, 0);

    ~step = ~step + 1;

    ~player_held[pitch] = false;

    // post the current note
    [\player, dt, inst, pitch, 0].postln;

}, srcID:~controller);

MIDIdef.noteOn(\input_on, {
    arg vel, pitch, chan, src;
    var perftime = Process.elapsedTime;
    var t2, dt;
    var inst = ~player_inst;
    var port_chan = ~prog2portchan.(inst);

    // ~m2.noteOn(12, pitch, 100);

    // mark time of current note
    t2 = Process.elapsedTime;
    dt = t2-(t?(t2-~delay));
    t = t2;
    ~player_t = t;

    // cancel any pending events
    ~clear_pending_events.();
    // ~pending_predictions.postln;

    ~player_sound.if{
        // release the previous note (if not properly released by noteoff)
        ~synths[pitch]!?(_.release(0.05));

        // play the current note
        ~use_synth.if{
            ~synths[pitch] = Synth(\pluck, [
                \freq, pitch.midicps, \vel, vel/127]);
        };
        // send MIDI
        port_chan[\port].noteOn(port_chan[\chan], pitch, vel);
    };
    (Process.elapsedTime - perftime).postln;

    //get a new prediction in light of current note
    ~predict_event.(inst, pitch, dt, vel);

    ~step = ~step + 1;

    ~player_held[pitch] = true;

    // post the current note
    [\player, dt, inst, pitch, vel].postln;
}, srcID:~controller);

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

    // there may be multiple outstanding queries to the model. when there are, it means
    // the earlier prediction is already voided, so it should be dropped here.
    var gate_event = (
        ~gate
        // && (~duet && (inst==~player_inst)).not
        && censor.not
        && (~pending_predictions==1)
        && (end==0)
    );

    ~pending_predictions = ~pending_predictions-1;
    gate_event.if{
        // cancel any pending events
        // since they are voided by the new prediction
        ~clear_pending_events.();
        // schedule the predicted note
        ~pending_events.add(Routine{
            var t2, dt_actual;
            // wait until predicted time
            // TODO: this could be made exact by passing timestamps thru OSC?
            (dt-~delay).max(0).sleep;

            // get the actual time since last event
            t2 = Process.elapsedTime;
            dt_actual = t2 - t;
            // mark the actual time of current note
            t = t2;
            ~machine_t = t;

            // play the current note
            ~synths[pitch]!?(_.release(0.05));
            // send MIDI
            port_chan[\port].noteOff(port_chan[\chan], pitch, 64);
            (vel > 0).if{
                ~use_synth.if{
                    ~synths[pitch] = Synth(\pluck, [
                        \freq, pitch.midicps, \vel, vel/127]);
                };
                // send MIDI
                port_chan[\port].noteOn(port_chan[\chan], pitch, vel)
            }{
                ~synths[pitch] = nil;
            };

            // feed model its own prediction as input
            ~predict_event.(inst, pitch, dt_actual, vel);

            // crudely draw note on piano GUI
            /*~gui.if{
            AppClock.sched(0,{k.keyDown(pitch)});
            AppClock.sched(0.2,{k.keyUp(pitch)});
            }*/
            // post the current note
            [\model, step, dt, inst, pitch, vel].postln;

            ~step = ~step+1;
            // [\late, dt_actual-dt].postln;
            nil;
        }.play(SystemClock));
    };
    (end==1).if{
        // b.sendMsg("/predictor/reset");
        //release the last note
        ~synths.do(_.release(1.0));
        ~release_all.(127);
        // unset time so next note will have dt=0
        // t = nil;
        // \reset.postln
        \end.postln;
    };
}, "/prediction", nil);
)

(
// play the first note and let the model take over
t = nil;
~gate = false;
~duet = false;
~allow_end = true;
~instrument_temp = nil;//0.95;
~pitch_temp = nil;//0.9;
~rhythm_temp = nil;//1;
~timing_temp = nil;//0.1;
~min_time = nil;//~delay;
~velocity_temp = nil;//0;
~include_pitch = (24..48)*2;//nil
~clear_pending_events.();
~synths.do(_.release(1.0));
~release_all.(0);
b.sendMsg("/predictor/reset");
~player_inst = 128.rand+1;
{~gate = true; MIDIdef.all[\input_on].func.value(127.rand+1, 60+12.rand)}.defer(0.2);
// ~player_inst = 272;
// {~gate = true; MIDIdef.all[\input_on].func.value(128.rand+1, 27.rand+60)}.defer(0.2);
)

(
// reserve an instrument for the player
t = nil;
~seq!?(_.stop);
~gate = false;
~duet = true;
~allow_end = false;
~include_pitch = nil;
~instrument_temp = 1;
~pitch_temp = 1;
~rhythm_temp = 1;
~timing_temp = 0.1;
~velocity_temp = 1;
~clear_pending_events.();
~release_all.(0);
b.sendMsg("/predictor/reset");
~player_inst = 272;
s = 2;
{/*~seq = Routine{
    loop{
        MIDIdef.all[\input_on].func.value(100, 51);
        MIDIdef.all[\input_on].func.value(100, 36);
        (0.12*s).sleep;
        MIDIdef.all[\input_off].func.value(0, 51);
        (0.005*s).sleep;
        MIDIdef.all[\input_on].func.value(100, 51);
        (0.125*s).sleep;
        MIDIdef.all[\input_on].func.value(100, 46);
        (0.22*s).sleep;
        MIDIdef.all[\input_off].func.value(0, 36);
        (0.01*s).sleep;
        MIDIdef.all[\input_off].func.value(0, 46);
        (0.02*s).sleep;
    }
}.play(SystemClock);
{ ~gate=true }.defer(4);*/
}.defer(0.5);
)

(
~gate = false;
~seq!?(_.stop);
~clear_pending_events.();
~release_all.(0);
)