// neural harmonizer
// model answers each note from the controller with a simultaneous note

// TODO: multi-instrument vs same-instrument mode
// TODO: enque simultaneous events / block waiting for response
// TODO: triads+
// TODO: above/below modes
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
MIDIdef.noteOn(\input_on, {
    arg vel, pitch, chan, src;
    var perftime = Process.elapsedTime;
    ~m2.noteOn(12, pitch, 100);
    (Process.elapsedTime - perftime).postln;
}, srcID:~controller);
)

(
// map instruments to channels + buses for the Ableton set
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
);
)

// set the delay for more precise timing
~delay = 0.01;

// duet with the model
// feeds the model's predictions back to it as well as player input
(
~velocity_temp = 1;
~pitch_temp = 0.9;

~player_inst = 1;
~use_synth = false;
~exclude_pitch = nil;
~model_inst = 258;

~gate = true;

~synths = nil!128;
~player_held = Dictionary.new; // maps player pitch to harmonizer pitch

~pending_predictions = List.new; // queue of player pitches for which harmonization is pending

// ~step = 0;

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

~feed_event = {
    arg inst, pitch, dt, vel;
    b.sendMsg("/predictor/feed",
        \inst, inst, \pitch, pitch, \time, dt, \vel, vel);
};

~predict_event = {
    arg inst, pitch, dt, vel;
    var kw = List[
        \inst, inst, \pitch, pitch, \time, dt, \vel, vel,
        \fix_time, 0, \min_vel, 1, \fix_instrument, ~model_inst,
        \allow_end, false,
        \exclude_pitch, ~player_held.values
    ];
    // ~duet.if{kw.add(\exclude_instrument); kw.add(~player_inst)};
    ~global_kws.(kw, [
        // \min_time, \max_time,
        // \instrument_temp,
        \pitch_temp,
        // \rhythm_temp, \timing_temp,
        \velocity_temp,
        // \exclude_pitch
    ]);

    b.sendMsg("/predictor/predict", *kw);
    ~pending_predictions.addFirst(pitch);

};

// ~clear_pending_events = {~pending_events.do{arg pe; pe.stop}.clear};

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
    var port_chan = ~prog2portchan.(~player_inst);
    var harm_port_chan = ~prog2portchan.(~model_inst);
    var harm_pitch = ~player_held.removeAt(pitch);

    // mark time of current note
    t2 = Process.elapsedTime;
    dt = t2-(t?(t2-~delay));
    t = t2;
    ~player_t = t;

    // release the previous note
    // ~synths[pitch]!?(_.release(0.05));
    // ~synths[pitch] = nil;

    // send MIDI
    port_chan[\port].noteOff(port_chan[\chan], pitch);

    // inform model of player release
    ~feed_event.(~player_inst, pitch, dt, 0);

    // release harmonized note
    harm_pitch.isKindOf(Number).if{
        harm_port_chan[\port].noteOff(
            harm_port_chan[\chan], harm_pitch)
    };

    // inform model of harmonizing release
    ~feed_event.(~model_inst, harm_pitch, 0, 0);

    // post the current note
    [\player_off, dt, ~player_inst, pitch, 0].postln;
    [\model_off, 0, ~model_inst, harm_pitch, 0].postln;

}, srcID:~controller);

MIDIdef.noteOn(\input_on, {
    arg vel, pitch, chan, src;
    // var perftime = Process.elapsedTime;
    var t2, dt;
    var inst = ~player_inst;
    var port_chan = ~prog2portchan.(inst);

    // mark time of current note
    t2 = Process.elapsedTime;
    dt = t2-(t?(t2-~delay));
    t = t2;
    ~player_t = t;

    /*// release the previous note (if not properly released by noteoff)
    ~synths[pitch]!?(_.release(0.05));
    // play the current note
    ~use_synth.if{
        ~synths[pitch] = Synth(\pluck, [
            \freq, pitch.midicps, \vel, vel/127]);
    };*/
    // send MIDI
    port_chan[\port].noteOn(port_chan[\chan], pitch, vel);

    //get a harmonization in light of current note
    ~predict_event.(inst, pitch, dt, vel);

    // ~step = ~step + 1;

    ~player_held[pitch] = \pending;

    // post the current note
    [\player_on, dt, inst, pitch, vel].postln;
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

    // there may be multiple outstanding queries to the model. when there are, it means
    // the earlier prediction is already voided, so it should be dropped here.
    var gate_event = (
        ~gate
    );

    //get the player pitch which this is harmonizing
    var player_pitch = ~pending_predictions.pop;

    gate_event.if{
        // play the harmonizing note
        var t2, dt_actual;

        // get the actual time since last event
        t2 = Process.elapsedTime;
        dt_actual = t2 - t;
        // mark the actual time of current note
        t = t2;
        ~machine_t = t;

        // play the harmonizing note
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

        ~player_held[player_pitch] = pitch;

        // feed model
        ~feed_event.(inst, pitch, dt_actual, vel);

        // post the current note
        [\model, dt, inst, pitch, vel].postln;
    };
}, "/prediction", nil);
)

(
// reset
t = nil;
~gate = false;
~pitch_temp = nil;//0.9;
~velocity_temp = nil;//0;
~synths.do(_.release(1.0));
~pending_predictions.clear;
~player_held.clear;
~release_all.(0);
b.sendMsg("/predictor/reset");
~player_inst = 257;
~model_inst = 258;
// ~player_inst = 128.rand+1;
// ~model_inst = 128.rand+1;
// ~model_inst = 129;
{~gate = true}.defer(0.2);
)

(
~gate = false;
~seq!?(_.stop);
~clear_pending_events.();
~release_all.(0);
)