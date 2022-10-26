
MRP {
	var osc;
	var midich=0;
	var midikeyboardview, guiFlag = false;

	*new { |ip="127.0.0.1", port=57120|
		^super.new.initMRP(ip, port);
	}

	initMRP { |ip, port|
		osc = NetAddr(ip, port); // what is the MRP address?
		MIDIClient.init;
		MIDIIn.connectAll;
		this.defineMIDIdefs;
	}

	defineMIDIdefs {
		// TODO: complete MIDI support

		MIDIdef.noteOn(\mrp_noteon, {arg ...args;
			args.postln;
			midich = midich + 1 % 16;
			this.noteOn(midich, args[1], args[0]);
			if(guiFlag, { {midikeyboardview.keyDown(args[1])}.defer });
		});

		MIDIdef.noteOff(\mrp_noteoff, {arg ...args;
			args.postln;
			this.noteOn(midich, args[1], args[0]);
			if(guiFlag, { {midikeyboardview.keyUp(args[1])}.defer });
		});
	}

	noteOn { |channel, note, vel|
		osc.sendMsg("/mrp/midi", channel, note, vel);
	}

	noteOff { |channel, note|
		osc.sendMsg("/mrp/midi", channel, note, 0); // Thor guessing that midi vel 0 = noteOff
	}

	// brightness is an independent map to harmonic content, reduced to a linear scale
	brightness{ |channel, note, val|
		osc.sendMsg("/mrp/quality/brightness", channel, note, val);
	}

	// intensity is a map to amplitude and harmonic content, relative to the current intensity
	intensity{ |channel, note, val|
		osc.sendMsg("/mrp/quality/intensity", channel, note, val);
	}

	// Frequency base is relative to the fundamental frequency of the MIDI note
	pitch{ |channel, note, val|
		osc.sendMsg("/mrp/quality/pitch", channel, note, val);
	}

	// Frequency vibrato is a periodic modulation in frequency, zero-centered (+/-1 maps to range loaded from XML)
	vibrato{ |channel, note, val|
		osc.sendMsg("/mrp/quality/pitch/vibrato", channel, note, val);
	}

	harmonic{ |channel, note, val|
		osc.sendMsg("/mrp/quality/harmonic", channel, note, val);
	}

	// change damper value
	pedaldamper { |val|
		osc.sendMsg("/mrp/pedal/damper", val);
	}

	// change sostenuto value
	pedalsostenuto { |val|
		osc.sendMsg("/mrp/pedal/sostenuto", val);
	}

	allNotesOff { |val|
		osc.sendMsg("/mrp/ui/allnotesoff", val);
	}

	createGUI {
        var win, midiclientmenu, amenu, anothermenu;
		var bounds = Rect(20, 5, 1200, 370);

		guiFlag = true;

		win = Window.new("- MRP GUI -", Rect(100, 500, bounds.width+20, bounds.height+10), resizable:false).front;
		win.alwaysOnTop = true;
		midikeyboardview = MIDIKeyboard.new(win, Rect(10, 70, 990, 160), 5, 36)
				.keyDownAction_({arg key;
			        midich =  midich + 1 % 16 ;
			        this.noteOn(midich, key, 60);
	                "Note ON :".post; key.postln;

				})
				.keyTrackAction_({arg key, x, y;
			        if(key.isNil.not, {
				        midich =  midich + 1 % 16;
				        this.noteOn(midich, key, 60);
			        });
                	"Key TRACK :".post; [key, x, y].postln;
				})
				.keyUpAction_({arg key;
			        "Note OFF :".post; key.postln;
				     this.noteOff(midich, key);
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

m = MRP.new;
m.noteOn(1, 60, 80)
m.noteOff(1, 60);
m.pedaldamper(0.1);

m.createGUI();




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


