// in this example the model's predictions are fed back to it so it plays itself.
Quarks.install("API")


(
///////// create Notochord API object /////////
~noto = Notochord();
~noto.notochordPath = "/Users/victor/Downloads/notochord_lakh_20G.ckpt";
// ~noto.notochordPath = "/Users/victor/Downloads/1368.ckpt";
~noto.startNotochord; // run notochord in a terminal
)


(
///////// create FluidSynth MIDI output object /////////
~output = NotoFluidOutput();
~output.soundFontPath = "/Users/victor/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2";
~output.startFluidSynth; // run fluidsynth in a terminal
)


(
///////// define scheduler /////////
~player_instrument = nil;
~gate = true;

// track the current predicted next event
~predicted_event = nil;

~step = 0;
~last_event_time = nil;

~noto.reset;
~noto.dropOldQueries = true;

~clear_prediction = {
	~predicted_event!?(_.stop);
	~predicted_event = nil;
};

~hush = {
	~gate = false;
	~clear_prediction.();
	~output.releaseAll;
};

// play an event, feed it to Notochord and request a prediction
~event = {
    arg inst, pitch, vel;

    var event_time, dt;

    // mark time of current event
    event_time = Process.elapsedTime;
	dt = ~last_event_time.notNil.if{
		event_time-~last_event_time}{0};
    ~last_event_time = event_time;

    // cancel any pending events
	~clear_prediction.();

	// send MIDI
	~output.send(inst, pitch, vel);

    //get a new prediction in light of current note
	~exclude_instrument = ~player_instrument!?([_]);
    ~noto.feedQuery(
		\inst, inst, \pitch, pitch, \time, dt, \vel, vel,
		\handle, event_time, *~noto.globalArgs);

    ~step = ~step + 1;

	[\played, inst, pitch, vel].postln;
};


// MIDI from controller
~input = NotoInput("MPKmini2");
~input.noteOff{
    arg vel, pitch, chan, src;
	~player_inst!?(~event.(_, pitch, 0));
};
~input.noteOn{
	arg vel, pitch, chan, src;
	~player_inst!?(~event.(_, pitch, vel));
};

// OSC return from python
~noto.handler = { |args|
	var end = args[\end];
	// absolute time of last event according to notochord
	var last_event_notochord = args[\handle];

    // there may be multiple outstanding queries to the model.
	// when there are, it means
    // the earlier prediction is already voided, so it should be dropped
    var gate_event = (
        ~gate
        && (end==0)
    );

    gate_event.if{
        // schedule the predicted note
        ~predicted_event = Routine{
			// absolute time of this event
			var sched_time = last_event_notochord + args[\time];
			// actual delta from now to scheduled time
			var sleep_time = sched_time - Process.elapsedTime;

            // wait until predicted time
			(sleep_time<0).if{
				[\late, sleep_time].postln
			}{
				[\waiting, sleep_time].postln
			};
			sleep_time.max(0).sleep;

			// \prediction_reached.postln;

			// if the Routine gets here without being canceled,
			// the predicted event occurs
			// fork the event into to different Routine so it doesn't
			// cancel itself!
			Routine{
				~event.(args[\instrument], args[\pitch], args[\velocity])
			}.play(SystemClock);
			// and the Routine is not rescheduled
            nil
        }.play(SystemClock);
    };
	(end==1).if{
        //release the last note
        ~output.releaseAll(127);
        \end.postln;
    };
};
)









(
// play the first note and let the model take over
~gate = false;
~last_event_time = nil;

~player_inst = nil;
~include_instrument = nil;

~allow_end = true;
~instrument_temp = nil;
~pitch_temp = nil;
~rhythm_temp = nil;
~timing_temp = nil;
~min_time = nil;
~max_time = nil;
~velocity_temp = nil;
~include_pitch = nil;
~clear_prediction.();

~output.releaseAll;
~noto.reset;
~gate = true; ~noto.query(\handle, nil, *~noto.globalArgs);
)


~hush.()











(
// reduced sampling temperature,
// uniformly random first instrument
~gate = false;
~last_event_time = nil;

~player_inst = nil;
~include_instrument = nil;

~allow_end = true;
~instrument_temp = 0.97;
~pitch_temp = 0.9;
~rhythm_temp = 1;
~timing_temp = 0.1;
~min_time = 0.015;
~max_time = nil;
~velocity_temp = 0;
~include_pitch = nil;
~clear_prediction.();

~output.releaseAll;
~noto.reset;
~gate = true; ~event.(128.rand+1, 60+12.rand, 100)
)


~hush.()







(
// restricted instrumentation and pitch set
~gate = false;
~last_event_time = nil;

~player_inst = nil;
~include_instrument = [5, 16, 21, 29, 35, 69, 74];

~allow_end = false;
~instrument_temp = 1;
~pitch_temp = 1;
~rhythm_temp = 1;
~timing_temp = 0.1;
~min_time = 0.015;
~max_time = 0.125;
~velocity_temp = 1;
~include_pitch = ([[0,3,7,9]]+[40,52,64,76]).flatten;
~clear_prediction.();

~output.releaseAll;
~noto.reset;
~gate = true; ~noto.query(\handle, nil, *~noto.globalArgs);
)

~include_pitch = ~include_pitch+1;

~include_pitch = (84..96);

~include_pitch = (30..42);


~hush.()










(
~gate = false;
~last_event_time = nil;

// choose instruments for the player and Notochord
~player_inst = 21;
~include_instrument = [12];
~allow_end = true;

~instrument_temp = 1;
~pitch_temp = 1;//0.95;
~rhythm_temp = 1;
~timing_temp = 1;
~min_time = 0.015;
~max_time = nil;
~velocity_temp = 0;
~include_pitch = nil;

~clear_prediction.();
~output.releaseAll;
~noto.reset;
Routine{4.sleep; ~gate = true}.play;
)


~hush.()






