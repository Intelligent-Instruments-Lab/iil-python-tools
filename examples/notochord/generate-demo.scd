// in this example the model's predictions are fed back to it so it plays itself.
// an instrument can also be assigned to a MIDI input to play along with Notochord

Quarks.install("API")


(
///////// create Notochord API object /////////
~noto = Notochord();
// ~noto.notochordPath = "/Users/victor/Downloads/notochord_lakh_20G.ckpt";
// ~noto.notochordPath = "/Users/victor/notochord_lakh_23G.pt";
~noto.notochordPath = "/Users/victor/notochord_lakh_50G_deep.pt";
// ~noto.notochordPath = "/Users/victor/notochord_lakh_30G_deep.pt";
~noto.startNotochord; // run notochord in a terminal
)


(
///////// create MIDI input object /////////
// this is just a convenience, you could use MIDIIn or MIDIdef directly instead
// change this to the name of your MIDI controller
~controller = "MPKmini2";
// MIDI from controller
~input = NotoInput(~controller);
~input.noteOff{
    arg vel, pitch, chan, src;
	// play a note-off event into notochord on the player instrument
	~player_inst!?(~event.(_, pitch, 0));
};
~input.noteOn{
	arg vel, pitch, chan, src;
	// play a note-on event into notochord on the player instrument
	~player_inst!?(~event.(_, pitch, vel));
};
)


(
///////// create FluidSynth MIDI output object /////////
~output = NotoFluidOutput();
~output.soundFontPath = "/Users/victor/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2";
~output.startFluidSynth; // run fluidsynth in a terminal
)


(
///////// define scheduler /////////
~player_inst = nil;
~gate = true;

// track the current predicted next event
~predicted_event = nil;

~step = 0;
~last_event_time = nil;

~noto.reset;
~noto.dropOldQueries = true;

// cancel a predicted event before it happens
~clear_prediction = {
	~predicted_event!?(_.stop);
	~predicted_event = nil;
};

// stop playing predictions and release all notes
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
	~exclude_inst = ~player_inst!?([_]);
    ~noto.feedQuery(
		\inst, inst, \pitch, pitch, \time, dt, \vel, vel,
		\handle, event_time, *~noto.globalArgs);

    ~step = ~step + 1;

	[\played, inst, pitch, vel].postln;
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

			// if the Routine gets here without being canceled,
			// the predicted event occurs
			// fork the event into a different Routine so it doesn't
			// cancel itself!
			Routine{
				~event.(args[\inst], args[\pitch], args[\vel])
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
~include_inst = nil;

~allow_end = true;
~insrumentt_temp = nil;
~pitch_temp = nil;
~rhythm_temp = nil;
~timing_temp = nil;
~min_time = nil;
~max_time = nil;
~velocity_temp = nil;
~include_pitch = nil;
~clear_prediction.();
~allow_anon = false;

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
~include_inst = nil;

~allow_end = true;
~instrument_temp = 0.98;
~pitch_temp = 0.95;
~rhythm_temp = 1;
~timing_temp = 0.1;
~min_time = nil;//0.015;
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
~include_inst = [5, 16, 21, 29, 35, 69, 74];

~allow_end = false;
~instrument_temp = 1;
~pitch_temp = 1;
~rhythm_temp = 1;
~timing_temp = 0.1;
~min_time = nil;//0.015;
~max_time = nil;//0.125;
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
~player_inst = 57;
~include_inst = [61]; //notochord instruments
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
Routine{5.sleep; ~gate = true}.play;
)

~hush.()

/*
Piano

    1 Acoustic Grand Piano
    2 Bright Acoustic Piano
    3 Electric Grand Piano
    4 Honky-tonk Piano
    5 Electric Piano 1 (usually a Rhodes Piano)
    6 Electric Piano 2 (usually an FM piano patch)
    7 Harpsichord
    8 Clavinet

Chromatic Percussion

    9 Celesta
    10 Glockenspiel
    11 Music Box
    12 Vibraphone
    13 Marimba
    14 Xylophone
    15 Tubular Bells
    16 Dulcimer

Organ

    17 Drawbar Organ
    18 Percussive Organ
    19 Rock Organ
    20 Church Organ
    21 Reed Organ
    22 Accordion
    23 Harmonica
    24 Tango Accordion

Guitar

    25 Acoustic Guitar (nylon)
    26 Acoustic Guitar (steel)
    27 Electric Guitar (jazz)
    28 Electric Guitar (clean)
    29 Electric Guitar (muted)
    30 Electric Guitar (overdriven)
    31 Electric Guitar (distortion)
    32 Electric Guitar (harmonics)

Bass

    33 Acoustic Bass
    34 Electric Bass (finger)
    35 Electric Bass (picked)
    36 Fretless Bass
    37 Slap Bass 1
    38 Slap Bass 2
    39 Synth Bass 1
    40 Synth Bass 2

Strings

    41 Violin
    42 Viola
    43 Cello
    44 Contrabass
    45 Tremolo Strings
    46 Pizzicato Strings
    47 Orchestral Harp
    48 Timpani

Ensemble

    49 String Ensemble 1
    50 String Ensemble 2
    51 Synth Strings 1
    52 Synth Strings 2
    53 Choir Aahs
    54 Voice Oohs (or Doos)
    55 Synth Voice or Solo Vox
    56 Orchestra Hit

Brass

    57 Trumpet
    58 Trombone
    59 Tuba
    60 Muted Trumpet
    61 French Horn
    62 Brass Section
    63 Synth Brass 1
    64 Synth Brass 2

Reed

    65 Soprano Sax
    66 Alto Sax
    67 Tenor Sax
    68 Baritone Sax
    69 Oboe
    70 English Horn
    71 Bassoon
    72 Clarinet

Pipe

    73 Piccolo
    74 Flute
    75 Recorder
    76 Pan Flute
    77 Blown bottle
    78 Shakuhachi
    79 Whistle
    80 Ocarina

Synth Lead

    81 Lead 1 (square)
    82 Lead 2 (sawtooth)
    83 Lead 3 (calliope)
    84 Lead 4 (chiff)
    85 Lead 5 (charang, a guitar-like lead)
    86 Lead 6 (space voice)
    87 Lead 7 (fifths)
    88 Lead 8 (bass and lead)

Synth Pad

    89 Pad 1 (new age or fantasia, a warm pad stacked with a bell)
    90 Pad 2 (warm)
    91 Pad 3 (polysynth or poly)
    92 Pad 4 (choir)
    93 Pad 5 (bowed glass or bowed)
    94 Pad 6 (metallic)
    95 Pad 7 (halo)
    96 Pad 8 (sweep)

Synth Effects

    97 FX 1 (rain)
    98 FX 2 (soundtrack, a bright perfect fifth pad)
    99 FX 3 (crystal)
    100 FX 4 (atmosphere, usually a nylon-like sound)
    101 FX 5 (brightness)
    102 FX 6 (goblins)
    103 FX 7 (echoes or echo drops)
    104 FX 8 (sci-fi or star theme)

Ethnic

    105 Sitar
    106 Banjo
    107 Shamisen
    108 Koto
    109 Kalimba
    110 Bag pipe
    111 Fiddle
    112 Shanai

Percussive

    113 Tinkle Bell
    114 Agogô
    115 Steel Drums
    116 Woodblock
    117 Taiko Drum
    118 Melodic Tom or 808 Toms
    119 Synth Drum
    120 Reverse Cymbal

Sound Effects

    121 Guitar Fret Noise
    122 Breath Noise
    123 Seashore
    124 Bird Tweet
    125 Telephone Ring
    126 Helicopter
    127 Applause
    128 Gunshot*/

