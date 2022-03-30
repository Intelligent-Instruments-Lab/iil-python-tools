// in this example the Linnstrument displays an interface to perform pitches
// by their likelihood under the NotePredictor model instead of by MIDI number

// the grid in the upper left gives control of pitches from
// the single most likely (cyan) to least likely (pink).

// the white pad samples pitches stochastically from the model.
// the lone yellow pad resets the model.

// model predictions are conditioned on performed timing and velocity.

(
~use_linn = true; // use linnstrument
~gui = false; // use keyboard GUI
MIDIIn.connectAll;
b = NetAddr.new("127.0.0.1", 9999);
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
// Server.default.options.inDevice_("mic-buds");
// Server.default.options.outDevice_("mic-buds");
~gui.if{
    k = MIDIKeyboard.new(bounds: Rect(0, 0, 500, 100), octaves:11, startnote:0)
};
~linn_reset = {
    ~linn.allLightsOff;
    ~linn.setNoteOnAction({arg x,y,degree,freq,amp;
        var idx;
        [x,y,amp].postln;
        (y<5).if{
            idx = case
            {y==0}{x}
            {y==1}{x*2+5}
            {y==2}{x*4+16}
            {y==3}{x*8+40}
            {y==4}{127-x};
            idx.postln;
            ~midi_handle.(amp*127, idx);
        }{
            case
            {x==0}{
                \sample.postln;
                ~midi_handle.(amp*127, nil);}
            {x==2}{
                \reset.postln;
                ~model_reset.()
            }
        }
    });
    ~linn.lightOn(0,0,4); //0
    4.do{arg x; ~linn.lightOn(x+1,0,3)}; //1-4
    5.do{arg x; ~linn.lightOn(x, 1, 10)}; //5-13
    5.do{arg x; ~linn.lightOn(x, 2, 2)}; //16-32
    5.do{arg x; ~linn.lightOn(x, 3, 9)}; //40-72
    ~linn.lightOn(0,4,11); //127
    4.do{arg x; ~linn.lightOn(x+1,4,1)}; //126-123

    ~linn.lightOn(0, 6, 8); // sample
    ~linn.lightOn(2, 6, 2); // reset


};
s.waitForBoot{
    ~use_linn.if{
        ~linn = IILinnstrument.new(nil);
        ~linn_reset.();
    }
};
)

OSCdef.trace(false)
// ~linn_reset.()

(
SynthDef(\pluck, {
    var vel = \vel.kr;
    var signal = Saw.ar(\freq.kr(20), 0.2) * EnvGate.new(1);
    var fr = 2.pow(Decay.ar(Impulse.ar(0), 3)*6*vel+8);
    signal = BLowPass.ar(signal, fr)*vel;
    Out.ar([0,1], signal);
}).add
)

// ~linn.setNoteOnAction({}); ~linn.setNoteOffAction({});


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

~model_reset = {
    ~last_pitch = nil;
    ~last_dt = nil;
    ~last_vel = nil;
    t = Process.elapsedTime;
    b.sendMsg("/predictor/reset");
    y!?{y.free};
    y = nil;
    b.sendMsg("/predictor/predict", \pitch, 128, \time, 0, \vel, 0);

};

~model_reset.();

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
~midi_handle = {
    arg vel, pitch, chan, src;
    var t2 = Process.elapsedTime;
    var dt = t2-(t?t2); //time since last note

    // release the previous note
    y.release(0.1);

    // attack the current note with the old pitch
    y = Synth(\pluck, [\freq, ~last_pitch.midicps, \vel, vel/127]);

    // get a new prediction in light of last note,
    // fixing dt and vel to performed values so just pitch is predicted
    pitch.notNil.if{
        b.sendMsg("/predictor/predict",
            \pitch, ~last_pitch, \time, ~last_dt, \vel, ~last_vel,
            \index_pitch, pitch, \fix_time, dt, \fix_vel, vel);
    }{
        b.sendMsg("/predictor/predict",
            \pitch, ~last_pitch, \time, ~last_dt, \vel, ~last_vel,
            \fix_time, dt, \fix_vel, vel);
    };

    ~last_dt = dt;
    ~last_vel = vel;

    // mark time of current note
    t = t2;
};

~use_linn.not.if{
    MIDIdef.noteOn(\input, ~midi_handle);
};

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

~model_reset.()
// send a note manually if you don't have a midi controller
MIDIdef.all[\input].func.(64, 16) //velocity, "pitch"

// load another model
// b.sendMsg("/predictor/load", "/path/to/checkpoint");