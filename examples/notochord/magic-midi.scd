// neural harmonizer
// model answers each note from the controller with a simultaneous note

// TODO: multi-instrument vs same-instrument mode
// TODO: enqueue simultaneous events / block waiting for response
// TODO: triads+
// TODO: above/below modes

(
///////// create Notochord API object /////////
~noto = Notochord();
~noto.notochordPath = "/Users/victor/notochord_lakh_50G_deep.pt";
~noto.startNotochord; // run notochord in a terminal
)


(
// run fluidsynth in a terminal
NotoFluidOutput.startFluidSynth(soundFontPath: "/Users/victor/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2");
)
/*
(
///////// create FluidSynth MIDI output object /////////
~output = NotoFluidOutput();
)*/


(
~noto.isNil.if{~noto = Notochord()};
~output.isNil.if{~output = NotoFluidOutput()};
///////// define scheduler /////////
~player_held = IdentityDictionary.new; // maps player notes to harmonizer notes

~last_event_time = nil;

~noto.reset;
~noto.dropOldQueries = false;

~hush = {
	~output.releaseAll;
};

~play_event = {
	arg pitch, vel, inst=nil, chan=nil;
	var event_time = Process.elapsedTime;
	var handle;
	var dt = ~last_event_time.notNil.if{
		event_time-~last_event_time}{0};
	~last_event_time = event_time;

	// auto reset
	(dt > 10).if{
		~noto.reset;
	};

	chan.notNil.if{
		handle = ~get_handle.(chan, pitch);
		["querying for completion:", handle].postln;
		//queryFeed for complete noteOn
		//(which will be played when it returns from Notochord)
		~player_held.add(handle -> \pending);
		~include_inst = (chan==1).if{(129..256)}{(1..128)};
		~noto.queryFeed(
			\next_time, dt, \next_vel, vel,
			\include_pitch, (pitch-1..pitch+1),
			*~noto.globalArgs
		).then{ |res|
			["updating event with completion:", res].postln;
			~output.send(res[\inst], res[\pitch], res[\vel]);
			// store the inst and pitch for later ending it
			~player_held.add(handle -> [res[\inst], res[\pitch]]);
		};
	}{
		~noto.feed(\inst, inst, \pitch, pitch, \time, dt, \vel, vel);
		//play the note
		~output.send(inst, pitch, vel);
	};

};

~get_handle = {
	arg chan, pitch;
	(chan.asString++"_"++pitch.asString).asSymbol;
};

// MIDI from controller
~input = NotoInput("MPKmini2");

~input.noteOn{
    arg vel, pitch, chan, src;
	//play the note on
	~play_event.(pitch, vel, chan:chan);
};

~input.noteOff{
	arg vel, pitch, chan, src;
	var handle = ~get_handle.(chan, pitch);
	var inst_pitch;

	inst_pitch = ~player_held.removeAt(handle);
	// play the noteOff
	// NOTE: it is possible for a noteOff to occur
	// *before* the completed noteOn has returned from Notochord
	(inst_pitch==\pending).if{
		"noteOff occurred before harmonizing NoteOn".postln;
	}{
		inst_pitch.notNil.if{
			var hinst = inst_pitch[0];
			var hpitch = inst_pitch[1];
			~play_event.(hpitch, 0, inst:hinst)
		}{
			"WARNING: noteOff handle not present".postln;
			// i think this can happen if notochord produces a double
			// noteOn
			handle.postln;
			~player_held.postln;
		}
	};
};

)

(
// ~output.releaseAll;
// ~noto.reset;
Routine{
	var key = 40+30.rand;
	~input.noteOnFn.(50, key, 0, nil);
	2.sleep;
	~input.noteOffFn.(0, key, 0, nil);
	nil
}.play
)

(
// reset
~last_event_time = nil;
~pitch_temp = 0.9;
~player_held.clear;

~output.releaseAll;
~noto.reset;
)


~hush.()