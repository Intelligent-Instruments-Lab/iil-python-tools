// in this example, the most likely next pitches (according to Notochord)
// are drawn on the Linnstrument as you play

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
    ~linn.setNoteOnAction({arg x,y,degree,freq,amp,midinote;
        var t2 = Process.elapsedTime;
        var dt = t2-(t?t2); //time since last note

        // release the previous note
        ~synth!?(_.release(0.1));

        // attack the current note
        ~synth = Synth(\pluck, [\freq, freq, \vel, amp]);

        //
        b.sendMsg("/predictor/predict",
            \pitch, midinote, \time, dt, \vel, amp*127,
            \pitch_topk, 5);

        // mark time of current note
        t = t2;
    });
};
s.waitForBoot{
    ~use_linn.if{
        ~linn = IILinnstrument.new(nil);
        ~linn_reset.();
    };
    SynthDef(\pluck, {
        var vel = \vel.kr;
        var signal = Saw.ar(\freq.kr(20), 0.2) * EnvGate.new(1);
        var fr = 2.pow(Decay.ar(Impulse.ar(0), 3)*6*vel+8);
        signal = BLowPass.ar(signal, fr)*vel;
        Out.ar([0,1], signal);
    }).add
};
)

(
// set the delay for more precise timing
~delay = 0.2;

~model_reset = {
    t = Process.elapsedTime;
    b.sendMsg("/predictor/reset");
    // ~synth!?{~synth.free};
    s.freeAll;
    ~synth = nil;
    b.sendMsg("/predictor/predict",
        \pitch, 128, \time, 0, \vel, 0,
        \pitch_topk, 5);

};
~model_reset.();
~linn_reset.();

// OSC return from python
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    var pitches, dts;
    var last_pitch = nil;
    var colormap = [8,3,10,9,1];
    var color = 0;

    // OSC arrays are not handled automatically by supercollider
    var stack = List[List[]];
    msg.do{arg item;
        case
        {item==$[}{stack.add(List[])}
        {item==$]}{var tmp = stack.pop; stack.last.add(tmp)}
        {true}{stack.last.add(item)};
    };
    msg = stack[0];

    pitches = msg[1];
    dts = msg[2];

    ~linn.allLightsOff;
    pitches.do{arg pitch;
        8.do{arg row;
            ~linn.lightOnMIDI(pitch,row,colormap[color]);
            // [pitch, row].postln
        };
        color = color + 1 % colormap.size;
    };

    msg.postln;
}, "/prediction", nil);
)



// ======
// additional useful blocks

~model_reset.()
// send a note manually if you don't have a midi controller
MIDIdef.all[\input].func.(64, 16) //velocity, "pitch"

// measure round-trip latency
(
OSCdef(\return, {
    arg msg, time, addr, recvPort;
    (Process.elapsedTime - t).postln;
}, '/prediction', nil);
t = Process.elapsedTime;
b.sendMsg("/predictor/predict",
    \pitch, 60+12.rand, \time, 0, \vel, 0,
    \index_pitch, 0,
    \sweep_time, true,
    // \fix_time, 0,
    // \fix_vel, 0
);
)
