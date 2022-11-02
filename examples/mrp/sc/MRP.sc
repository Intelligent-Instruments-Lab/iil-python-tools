

/*
XiiQuarks.new
Questions:
- Is there a default value for brightness, intensity, pitch, etc.
- what is the range in all of those?

// et12 tuning octave
t = {|i| 2.pow(i/12)}!12

// tunings of all 8 octaves
a = ({|i|if(i==0,{f=i+1},{f=f*2});{|j|2.pow(j/12)*27.5*f}!12}!8).flatten[0..87]

// the 12 harmonics of each key of the piano
a = ({|i|if(i==0,{f=i+1},{f=f*2});{|j|{|k|(2.pow(j/12)*27.5*f)*(k+1)}!12}!12}!8).flatten[0..87]

Post << a

// with pitchshift
x = {|pitchshift=0|

	({|i|if(i==0,{f=i+1},{f=f*2});{|j|{|k|(2.pow(j/12)*(27.5+pitchshift.linlin(-1,1,-1.6, 1.6))*f)*(k+1)}!12}!12}!8).flatten[0..87];
}

x.(0)

x = {|pitchshift=0|
a = ({|i|if(i==0,{f=i+1},{f=f*2});{|j|2.pow(j/12)*27.5+pitchshift.linlin(-1,1,-1.6, 1.6)*f}!12}!8).flatten[0..87]
}

x.(0.5)

*/

MRP {
	var osc;
	var midich=0;
	var midikeyboardview, guiFlag = false;
	var settings, notes, activeNotes;

	*new { |ip="127.0.0.1", port=7770|
		^super.new.initMRP(ip, port);
	}

	initMRP { |ip, port|
		osc = NetAddr(ip, port); // MRP address is 7770 by default?

		settings =  ().put('voices', ().put('max', 10).put('rule', 'oldest')).put('channel', 15).put('range', ().put('start', 21).put('end', 108)).put('qualities_max', 1.0).put('qualities_min', -1.0).put('maxHarmonics', 36);

		notes = { arg i;
			().put('channel', settings.channel)
			.put('status', \note_off)
			.put('midi', ()
				.put('number', i+1)
				.put('velocity', 0) // not in use by MRP PLL synth
				.put('aftertouch_poly', 0) //  not in use by MRP PLL synth
				.put('aftertouch_channel', 0) //  not in use by MRP PLL synth
				.put('pitch_bend', 0) // # not in use by MRP PLL synth
			)
			.put('qualities', ()
				.put('brightness', 0)
				.put('intensity', 0)
				.put('pitch', 0)
				.put('pitch_vibrato', 0)
				.put('harmonic', 0)
				.put('harmonics_raw', [])
			)
		}!128; // Creating a dict for all the possible midi notes (ignoring piano range)
			//}!(settings.range.end-settings.range.start);

		activeNotes = [];
		MIDIClient.init;
		MIDIIn.connectAll;
		this.defineMIDIdefs;
	}

	defineMIDIdefs {
		// TODO: complete MIDI support

		MIDIdef.noteOn(\mrp_noteon, {arg ...args;
			args.postln;
			midich = midich + 1 % 16;
			this.noteOn(args[1], args[0]);
			if(guiFlag, { {midikeyboardview.keyDown(args[1])}.defer });
		});

		MIDIdef.noteOff(\mrp_noteoff, {arg ...args;
			args.postln;
			this.noteOff(args[1], args[0]);
			if(guiFlag, { {midikeyboardview.keyUp(args[1])}.defer });
		});
	}

	noteOn { |note, vel=1|
		if((notes[note].status == \note_off) && ((note >= settings.range.start) && (note <= settings.range.end)), {
			if(activeNotes.size < settings.voices.max, {
				notes[note].status = \note_on;
				activeNotes = activeNotes.insert(0, note); // adds a new note in the first slot
			}, {
				this.noteOff(activeNotes.pop); // removes the last index in the array and turns the note off
				activeNotes = activeNotes.insert(0, note); // adds a new note in the first slot
			});
			osc.sendMsg("/mrp/midi", 0x9F, note, vel);
		}, {
			"Note is already ON or out of keyboard range".postln;
		});
	}

	noteOff { |note|
		if((notes[note].status == \note_on) && ((note >= settings.range.start) && (note <= settings.range.end)), {
			notes[note].status = \note_off;
			osc.sendMsg("/mrp/midi", 0x8F, note, 0); // We leave 0 here in val for the MRP simulator
		});
	}

	// brightness is an independent map to harmonic content, reduced to a linear scale
	brightness { |note, val|
		if(notes[note].status == \note_on, {
			notes[note].qualities.brightness = val;
			osc.sendMsg("/mrp/quality/brightness", settings.channel, note, val.asFloat);
		});
	}

	// intensity is a map to amplitude and harmonic content, relative to the current intensity
	intensity { |note, val|
		if(notes[note].status == \note_on, {
			notes[note].qualities.intensity = val;
			osc.sendMsg("/mrp/quality/intensity", settings.channel, note, val.asFloat);
		});
	}

	// Frequency base is relative to the fundamental frequency of the MIDI note
	pitch { | note, val|
		if(notes[note].status == \note_on, {
			notes[note].qualities.pitch = val;
			osc.sendMsg("/mrp/quality/pitch", settings.channel, note, val.asFloat);
		});
	}

	// Frequency vibrato is a periodic modulation in frequency, zero-centered (+/-1 maps to range loaded from XML)
	vibrato { |note, val|
		if(notes[note].status == \note_on, {
			notes[note].qualities.vibrato = val;
			osc.sendMsg("/mrp/quality/pitch/vibrato", settings.channel, note, val.asFloat);
		});
	}

	harmonicSweep { |note, val|
		if(notes[note].status == \note_on, {
			notes[note].qualities.harmonic = val;
			osc.sendMsg("/mrp/quality/harmonic", settings.channel, note, val.asFloat);
		});
	}

	harmonic { |note, val, amp=1|
		var valarray;
		if(notes[note].status == \note_on, {
			valarray = {0}!(settings.maxHarmonics+1);
			valarray[val] = amp;
			// make an array out of value
			notes[note].qualities.harmonics_raw = valarray;
			osc.sendMsg("/mrp/quality/harmonics/raw", settings.channel, note, *valarray.asFloat);
		});
	}

	harmonicsraw { |note, valarray|
		if(notes[note].status == \note_on, {
			notes[note].qualities.harmonics_raw = valarray;
			osc.sendMsg("/mrp/quality/harmonics/raw", settings.channel, note, *valarray.asFloat);
		});
	}
	// change damper value
	pedaldamper { |val|
		osc.sendMsg("/mrp/pedal/damper", val);
	}

	// change sostenuto value
	pedalsostenuto { |val|
		osc.sendMsg("/mrp/pedal/sostenuto", val);
	}

	allNotesOff {
		notes.do({arg note; note.status = \note_off});
		osc.sendMsg("/mrp/ui/allnotesoff");
	}

	simulator_ {arg boolean; if( boolean==true, { this.startMRPSimulator }, { this.stopMRPSimulator }) }

	startMRPSimulator {
		Server.default.boot;

		osc = NetAddr("127.0.0.1", 57120); // we now listen on SC port
		notes.do({arg note; note.put(\simsynth, nil) }); // add a slot for an SC synth in dict


SynthDef(\mrp, {arg freq=440, vel=1, intensity=0.6, gate=1, brightness=0.8, harmonic=1, pitch=0, harmonics_raw(#[ 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 ]);
	var piano, pitchednote, lpf, env, dyn;
	piano = MdaPiano.ar(freq, 1, 100, decay: 10000, release: 1);
	pitchednote = PitchShift.ar(piano, 0.1, pitch.linlin(-1,1, 0.9405369056407, 1.0594630943593), 0, 0.004);
	lpf = RLPF.ar(pitchednote, freq * brightness.linlin(0,1,1,44), 0.6);
	dyn = DynKlank.ar(`[{|i|freq*(i+1)}!32, harmonics_raw, {0.1}!32], lpf*0.01);
	pitchednote = PitchShift.ar(dyn, 0.1, MouseX.kr(0,2), 0, 0.004);
	env = EnvGen.ar(Env.adsr(vel.linlin(0, 127, 3, 0.00000001), 0.3, 0.88, 1), gate, doneAction:2);
	Out.ar(0, Pan2.ar(dyn*env*intensity, 0));
}).add;

			/*
a =	Synth(\mrp, [\freq, 33.midicps, \vel, 1, \intensity, 1]);

a.set(\brightness, 0.1)
a.set(\intensity, 2)
a.set(\harmonics_raw, [0, 0, 0.2, 0.9, 0.9])
a.set(\harmonics_raw, {1.0.rand}!32)
a.set(\harmonics_raw, {[1,0].wchoose([0.2, 0.8])}!32)
a.set(\pitch, -1)
a.set(\pitch, 0)
a.set(\pitch, 1)


a =		Synth(\mrp, [\freq, 33.midicps, \vel, 1, \intensity, 1]);
a.set(\brightness, 0.15)
a.set(\harmonic, 2)
a.set(\harmonic, 6)
a.set(\intensity, 0.1)
a.set(\intensity, 0.2)
a.set(\intensity, 0.7)
*/


		OSCdef(\midi, {|msg, time, addr, recvPort|
			msg.postln;
			if(msg[3] != 0, {
				notes[msg[2]].simsynth = Synth(\mrp, [\freq, msg[2].midicps, \vel, msg[3]]);
			}, {
				notes[msg[2]].simsynth.release;
			});
		}, '/mrp/midi', osc);

		OSCdef(\brightness, {|msg, time, addr, recvPort|
			msg.postln;
			"brightness in simulation mode".postln;
			notes[msg[2]].simsynth.set(\brightness, msg[3]);
		}, '/mrp/quality/brightness', osc); // def style

		OSCdef(\harmonic, {|msg, time, addr, recvPort|
			msg.postln;
			"harmonic in simulation mode".postln;
			notes[msg[2]].simsynth.set(\harmonic, msg[3]);
		}, '/mrp/quality/harmonic', osc); // def style

		OSCdef(\intensity, {|msg, time, addr, recvPort|
			msg.postln;
			"intensity in simulation mode".postln;
			notes[msg[2]].simsynth.set(\intensity, msg[3]);
		}, '/mrp/quality/intensity', osc); // def style

	}

	stopMRPSimulator {
		osc = NetAddr("127.0.0.1", 7770); // back to MRP port
		OSCdef(\midi).free;  // unregister OSCdef
		OSCdef(\brightness).free;  // unregister OSCdef
		OSCdef(\harmonic).free;  // unregister OSCdef
		OSCdef(\intensity).free;  // unregister OSCdef
		// ... etc   TODO
	}

	createGUI {
        var win, midiclientmenu, amenu, anothermenu;
		var bounds = Rect(20, 5, 1200, 370);

		guiFlag = true;

		win = Window.new("- MRP GUI -", Rect(100, 500, bounds.width+20, bounds.height+10), resizable:false).front;
		win.alwaysOnTop = true;
		midikeyboardview = MIDIKeyboard.new(win, Rect(10, 70, 990, 160), 5, 36)
				.keyDownAction_({arg key;
			        this.noteOn(key, 60);
	                "Note ON :".post; key.postln;

				})
				.keyTrackAction_({arg key, x, y;
			        if(key.isNil.not, {
				        midich =  midich + 1 % 16;
				        this.noteOn(key, 60);
			        });
                	"Key TRACK :".post; [key, x, y].postln;
				})
				.keyUpAction_({arg key;
			        "Note OFF :".post; key.postln;
				     this.noteOff(key);
				});
       // midikeyboardview.keyTrackFlag = true;

		midiclientmenu = PopUpMenu.new(win,Rect(10,15,150,16))
				.font_(Font.new("Helvetica", 9))
				.canFocus_(false)
				.items_(MIDIClient.sources.collect({arg item; item.device + item.name}))
				.value_(0)
				.background_(Color.white)
				.action_({arg item;
					MIDIClient.sources.do({ |src, i| MIDIIn.disconnect(i, i) });
					MIDIIn.connect(item.value, MIDIClient.sources.at(item.value));
				});

		amenu = PopUpMenu.new(win,Rect(10,41,100,16))
				.font_(Font.new("Helvetica", 9))
                .items_([])
				.value_(0)
				.background_(Color.white)
				.action_({arg item;
					//this.synth_(synthdefs[item.value].asSymbol);
				});

		anothermenu = PopUpMenu.new(win,Rect(115,41,45,16))
				.font_(Font.new("Helvetica", 9))
				.items_({|i| ((i*2).asString++","+((i*2)+1).asString)}!26)
				.value_(0)
				.background_(Color.white)
				.canFocus_(false)
				.action_({arg item;
					//outbus = item.value * 2;
				});

		// TODO: create a close window function and set guiFlag to false;
		// TODO: add all kinds of sliders to test the behaviour of a string (control brightness etc)
	}

}


/*

testing MRP class

m = MRP.new("127.0.0.1", 7770); // 7770 is the MRP port I believe
m.noteOn(33, 1)
m.noteOff(33);
m.pedaldamper(0.1);
m.createGUI();


// testing the MRP SIMULATOR

m = MRP.new("127.0.0.1", 57120); // test sending OSC to SuperCollider
m.simulator = true
m.noteOn(36, 1)
m.brightness(36, 0.1)
m.brightness(36, 0.9)
m.intensity(36, 0.1)
m.intensity(36, 0.6)
m.harmonic(36, 2)
m.harmonic(36, 3)
m.harmonic(36, 4)
m.harmonic(36, 5)
m.harmonic(36, 6)
m.createGUI()

m.noteOff(36);





// test OSC def
OSCdef(\test, {|msg, time, addr, recvPort| msg.postln}, '/mrp/midi', n); // def style
OSCdef(\test, {|msg, time, addr, recvPort| msg.postln}, '/mrp/pedal/damper', n); // def style


*/



/*

// unused methods as to yet:

/ptrk/mute: [array int notes]
/ptrk/pitch: float freq, float amp
/quality/harmonics/raw: int midiChannel, int midiNote, [array of float harmonics] // ?
/ui/cal/save // Marked as LEGACY
/ui/cal/load // Marked as LEGACY

*/



// MRP

/*
/midi: byte a, byte b, byte c // standard 3-byte MIDI messages e.g. 144 90 60
/pedal/damper: int value || float value // change damper value
/pedal/sostenuto: int value || float value // change sostenuto value
/ptrk/mute: [array int notes]
/ptrk/pitch: float freq, float amp
/quality/brightness: int midiChannel, int midiNote, float brightness // brightness is an independent map to harmonic content, reduced to a linear scale
/quality/intensity: int midiChannel, int midiNote, float intensity // intensity is a map to amplitude and harmonic content, relative to the current intensity
/quality/pitch: int midiChannel, int midiNote, float pitch // Frequency base is relative to the fundamental frequency of the MIDI note
/quality/pitch/vibrato: int midiChannel, int midiNote, float pitch // Frequency vibrato is a periodic modulation in frequency, zero-centered (+/-1 maps to range loaded from XML)
/quality/harmonic: int midiChannel, int midiNote, float harmonic // ?
/quality/harmonics/raw: int midiChannel, int midiNote, [array of float harmonics] // ?
/ui/allnotesoff // turn all current notes off
/ui/cal/save // Marked as LEGACY
/ui/cal/load // Marked as LEGACY
/ui/cal/phase: float p
/ui/cal/volume: float v
/ui/cal/currentnote: int n // "Send a message announcing the current note, to update the UI"
/ui/gate // Marked as LEGACY
/ui/harmonic // Marked as LEGACY
/ui/patch/up // increment current program
/ui/patch/down // decrement current program
/ui/patch/set: int p // 0-N, set the current program to the given parameter
/ui/pianokey/calibrate/start
/ui/pianokey/calibrate/finish
/ui/pianokey/calibrate/abort
/ui/pianokey/calibrate/idle: // (?)
/ui/pianokey/calibrate/disable: [array int keys] // 0-127, disable specified keys
/ui/pianokey/calibrate/save: // only saves to `mrp-pb-calibration.txt`
/ui/pianokey/calibrate/load: // only loads from `mrp-pb-calibration.txt`
/ui/pianokey/calibrate/clear
/ui/status/keyboard: // get real-time calibration status (?)
/ui/tuning/global // Marked as LEGACY
/ui/tuning/stretch // Marked as LEGACY
/ui/volume: float vol // 0-1, >0.5 ? 4^((vol-0.5)/0.5) : 10^((vol-0.5)/0.5)
/ui/volume/raw: float vol // 0-1, set volume directly

*/


