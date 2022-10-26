// in this example the Linnstrument displays an interface to perform pitches
// by their likelihood under the NotePredictor model instead of by MIDI number

// the grid in the upper left gives control of pitches from
// the single most likely (cyan) to less likely (pink).

// the white pad samples pitches stochastically from the model.
// the lone yellow pad resets the model.

// model predictions are conditioned on performed timing and velocity.

Quarks.install("API")
//depends: IILinnstrument https://github.com/Intelligent-Instruments-Lab/IILinnstrument


(
~noto = Notochord();
// ~noto.notochordCmd; // show the current notochord terminal command
~noto.notochordPath = "/Users/victor/notochord_lakh_50G_deep.pt";
~noto.startNotochord; // runs notochord for you in a terminal
)


(
~synths = ();
~pitches = ();

~handle_pad = {
	arg x,y,vel;
	var key = (x.asString++"_"++y.asString).asSymbol;
	var idx;
	// [x,y,vel].postln;
	(y<7).if{
		idx = y;
		[\index, idx].postln;
		~midi_handle.(vel, idx, key);
	}{
		case
		{x==0}{
			\sample.postln;
			~midi_handle.(vel, nil, key);}
		{x==2}{
			\reset.postln;
			~model_reset.()
		}
	};
};

MIDIIn.connectAll;
~linn_reset = {
    ~linn.allLightsOff;
    ~linn.setNoteOnAction({arg x,y,degree,freq,amp;
		~handle_pad.(x,y,amp*127);
    });
	~linn.setNoteOffAction({arg x,y,degree,freq,amp;
		~handle_pad.(x,y,0);
    });

	// display surface
	4.do{arg x; ~linn.lightOn(x, 0, 4)};
    4.do{arg x; ~linn.lightOn(x, 1, 3)};
    4.do{arg x; ~linn.lightOn(x, 2, 10)};
    4.do{arg x; ~linn.lightOn(x, 3, 2)};
    4.do{arg x; ~linn.lightOn(x, 4, 9)};
    4.do{arg x; ~linn.lightOn(x, 5, 1)};
	4.do{arg x; ~linn.lightOn(x, 6, 11)};

    ~linn.lightOn(0, 7, 8); // sample
    ~linn.lightOn(2, 7, 2); // reset

};
s.waitForBoot{
	~linn = IILinnstrument.new(nil);
	~linn_reset.();
	SynthDef(\pluck, {
		var vel = \vel.kr;
		var env = Decay.ar(Impulse.ar(0), 3);
		var signal = (
			PinkNoise.ar*0.05*env + Saw.ar(\freq.kr(20), 0.2)
		) * EnvGate.new(1);
		var fr = 2.pow(env*6*vel+8);
		signal = BLowPass.ar(signal, fr)*vel;
		Out.ar([0,1], signal*0.2);
	}).add
};

~instrument = 257;

~model_reset = {
    ~last_event_time = nil;
	~noto.reset;
};

~model_reset.();

// MIDI from controller
~midi_handle = {
    arg vel, idx, key;
    var event_time = Process.elapsedTime;
	var dt = ~last_event_time.notNil.if{event_time-~last_event_time}{0};
	~last_event_time = event_time;

	(vel==0).if{
		// release synth for this key
		~synths[key].release(0.3);
		// feed the note-off
		~noto.feed(
			\inst, ~instrument, \pitch, ~pitches[key], \vel, 0, \time, dt
		);
	}{
		var pitch_args = idx.notNil.if{[\index_pitch, idx]}{[]};
		// attack the current note without a pitch
		~synths.add(key -> Synth(\pluck, [\vel, vel/127]));
		// query for pitch
		~noto.queryFeed(
			\next_inst, ~instrument,
			\next_vel, vel, \next_time, dt,
			\handle, key, *(~noto.globalArgs++pitch_args)
		)
	}
};
// OSC return from python
~noto.handler = {
    arg args;
	var handle = args[\handle].asSymbol;
	var pitch = args[\pitch];

	~synths[handle].set(\freq, pitch.midicps);
	~pitches.add(handle->pitch);
};
)