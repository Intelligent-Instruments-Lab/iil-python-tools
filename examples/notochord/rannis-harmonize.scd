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

(
///////// create FluidSynth MIDI output object /////////
~output = NotoFluidOutput();
)


(
///////// define scheduler /////////
~player_inst = 1;
~noto_inst = 1;

~player_held = IdentityDictionary.new; // maps player notes to harmonizer notes

~last_event_time = nil;

~noto.reset;
~noto.dropOldQueries = false;

~hush = {
	~output.releaseAll;
};

~play_event = {
	arg inst, pitch, vel, query=false, feed=false;
	var event_time = Process.elapsedTime;
	var handle = ~get_handle.(inst, pitch);
	var dt = ~last_event_time.notNil.if{
		event_time-~last_event_time}{0};
	~last_event_time = event_time;

	//play the note
	~output.send(inst, pitch, vel);

	query.if{
		//feedQueryFeed for harmonizing noteOn
		//(which will be played when it returns from Notochord)
		~player_held.add(handle -> \pending);
		~noto.feedQueryFeed(
			\inst, inst, \pitch, pitch, \time, dt, \vel, vel,
			\next_inst, ~noto_inst, \next_time, 0, \next_vel, vel,
			\handle, handle, *~noto.globalArgs)
	};
	feed.if{
		~noto.feed(\inst, inst, \pitch, pitch, \time, dt, \vel, vel)
	};

};

~get_handle = {
	arg inst, pitch;
	(inst.asString++"_"++pitch.asString).asSymbol;
};

// MIDI from controller
~input = NotoInput("MPKmini2");

~input.noteOn{
    arg vel, pitch, chan, src;
	//play the note on
	~play_event.(~player_inst, pitch, vel, query:true);
};

~input.noteOff{
	arg vel, pitch, chan, src;
	var handle = ~get_handle.(~player_inst, pitch);
	var hpitch;

	// play the noteOff
	~play_event.(~player_inst, pitch, 0, feed:true);

	hpitch = ~player_held.removeAt(handle);
	// play the harmonizing noteOff
	// NOTE: it is possible for a noteOff to occur
	// *before* the harmonizing note has returned from Notochord
	(hpitch==\pending).if{
		"noteOff occurred before harmonizing NoteOn".postln;
	}{
		hpitch.notNil.if{
			~play_event.(~noto_inst, hpitch, 0, feed:true)
		}{
			\error.postln;
			handle.postln;
			~player_held.postln;
		}
	};
};


// returning queries from Notochord
~noto.handler = { |args|
	var pitch = args[\pitch];
	// args.postln;
	// start the harmonizing note
	~play_event.(args[\inst], pitch, args[\vel]);
	// store the pitch for later ending it
	~player_held.add(args[\handle].asSymbol -> pitch);
};

)

(
// reset
~last_event_time = nil;
~pitch_temp = 0.9;
~player_held.clear;

~output.releaseAll;
~noto.reset;

~player_inst = 25;
~noto_inst = 25;
)


~hush.()