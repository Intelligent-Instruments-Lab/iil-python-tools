SynthDef(\mrp, {arg freq=440, amp=0.3, gate=1;
    var piano, env;
    piano = MdaPiano.ar(freq, amp, decay: 100, release: 1);
    env = EnvGen.ar(Env.adsr(0.0001, 0.3, 0.88, 1), gate, doneAction:2);
    Out.ar(0, Pan2.ar(pianoenv, 0));
}).add;

/
a = Synth(\mrp, [\freq, 343])
a.release(1)
*/


(
~notes = {nil}!127;
MIDIIn.connectAll;

MIDIdef.noteOn(\noteOn, {|vel, num|
    ~notes[num] = Synth(\mrp, [\freq, num.midicps]);
});

MIDIdef.noteOff(\noteOff, {|vel, num|
    ~notes[num].set(\gate, 0);
});
)