// in this example the model's predictions are fed back to it so it plays itself.
// an instrument can also be assigned to a MIDI input to play along with Notochord

// this version allows Notochord a 'budget' of NoteOn events equal to the number
// played on the controller

// issue: the budget can grow very large if the player is playing chords
// issue: the budget does not grow while the player sustains notes

// might be better to measure 'recent player activity' rather than number of events?

Quarks.install("API")

(
///////// create Notochord API object /////////
~noto = Notochord();
~noto.notochordPath = "/Users/victor/notochord_lakh_50G_deep.pt";
~noto.startNotochord; // run notochord in a terminal
)


( //////// OUTPUT MIDI mappings ////////
~output = NotoMappingOutput("Kjartan", "Bluetooth (internal)");
// this will automatically restrict the instruments notochord can sample
// to the keys of instrumentMap, and send each instrument to the given MIDI channel
// (but make sure not to use any other instruments in tidal either)
// this example requires you to set up synths on MIDI channels 2,3,4, and 10
// where 10 has a drum rack.
// uncomment lines to add more channels, and change to first number to use
// different notochord instruments.
~output.instrumentMap = (
    // notochord instrument (1-320): midi channel (1-16)
	// 1: 1, //grand piano -> channel 1
	257: 1, //first anonymous instrument -> channel 1
	258: 2,
	259: 3,
	260: 4,
	261: 5,
	262: 6,
	263: 7,
	264: 8,
	265: 9,
	129: 10,
	266: 11,
	267: 12,
	268: 13,
	269: 14,
	270: 15,
	271: 16,
	// 8: 2, //clavinet -> channel 2
	// 13: 3, //marimba -> channel 3
	// 17: 4, //drawbar organ -> channel 4
	// 27: 5, //jazz guitar -> channel 5
    // 39: 6, //synth bass 1 -> channel 6
    // 47: 7, //harp -> channel 7
    // 54: 8, //voice oohs -> channel 8
    // 57: 9, //trumpet -> channel 9
	// 129:10, //standard drumkit -> channel 10 (drums)
    // 67:11, //tenor sax -> channel 11
    // 76:12, //pan flute -> channel 12
    // 82:13, //saw lead -> channel 13
    // 90:14, //warm pad -> channel 14
    // 9:15, //celesta -> channel 15
    // 11:16, //music box -> channel 16
);
// this will automatically restrict the pitches notochord can sample
// when using a drum instrument to the keys of drumMap
// this example is based on the default 808 rack in Ableton
~output.drumMap = (
    //general MIDI number (in Notochord): desired MIDI number (in DAW)
    36:36, //electric bass drum -> C1
    37:37, //side stick -> C#1
    40:38, //electric snare -> D1
    39:39, //hand clap -> D#1
    75:40, //clave -> E1
    45:41, //low tom -> F1
    42:42, //closed hat-> F#1
    47:43, //mid tom -> G1
    70:44, //maracas -> G#1
    50:45, //hi tom -> A1
    46:46, //open hat -> A#1
    64:47, //low conga -> B1
    63:48, //mid conga -> C2
    51:49, //ride cymbal 1 -> C#2
    62:50, //hi conga -> D2
    56:51, //cowbell -> D#2
);

~allow_anon = true; //you can use anonymous instrument IDs if you like
)

(
///////// create MIDI input object /////////
// this is just a convenience, you could use MIDIIn or MIDIdef directly instead
// change this to the name of your MIDI controller
// ~controller = "MPKmini2";
~controller = "Kjartan";
// MIDI from controller
~input = NotoInput(~controller);
MIDIdef.freeAll;
~input.noteOff{
    arg vel, pitch, chan, src;
	// play a note-off event into notochord on the player instrument
	(chan==0).if{~player_inst}{~player_drum}!?(~event.(_, pitch, 0));
};
~input.noteOn{
	arg vel, pitch, chan, src;
	~gate=true;
	~budget = ~budget + 1;
	// ~gate.not.if{Routine{2.sleep; ~gate = true}.play};
	// play a note-on event into notochord on the player instrument
	(chan==0).if{~player_inst}{~player_drum}!?(~event.(_, pitch, vel));
};
// )

/*MIDIdef.bend(\stick, {arg val, chan, src;
	~noto.reset;
	~hush.();
}, chan:0);
MIDIdef.cc(\stick2, {arg val, num, chan, src;
	(num==1).if{
		~noto.reset;
		~hush.();
	};
}, chan:0);*/

~budget = 0;

// (
~noto.isNil.if{~noto = Notochord()};
// ~output.isNil.if{~output = NotoFluidOutput()};

///////// define scheduler /////////
~player_inst = nil;
~player_drum = nil;
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

	// auto reset
	(dt > 10).if{
		~noto.reset;
	};

    // cancel any pending events
	~clear_prediction.();

	// send MIDI
	~output.send(inst, pitch, vel);

    //get a new prediction in light of current note
	// ~exclude_inst = [~player_inst, ~player_drum].removeEvery([nil]);
	// ~include_inst = ~noto_inst++~soft_gate.if{[~player_inst, ~player_drum]}{[]};
	~include_inst = ~noto_inst++[~player_inst, ~player_drum].reject(_.isNil);
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
	// stop if the global gate is closed or end is predicted
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
			var wait_time = sleep_time + 0.2;


			var query_new = {
				Routine{
					// TODO: this is ugly. should rethink globalArgs
					// to be overridden by explicit vals...
					var tmp_min_time = ~min_time;
					~min_time = wait_time;
					~noto.query(\handle, last_event_notochord, *~noto.globalArgs);
					~min_time = tmp_min_time;
				}.play(SystemClock)
			};

			~noto_inst.includes(args[\inst].postln).if{

				// wait until predicted time
				(sleep_time<0).if{
					[\late, sleep_time].postln
				}{
					[\waiting, sleep_time].postln
				};
				sleep_time.max(0).sleep;

				// if the Routine gets here without being canceled,
				// and there is budget,
				// the predicted event occurs
				// fork the event into a different Routine so it
				// doesn't cancel itself!
				((~budget > 0) || (args[\vel]==0)).if{
					(args[\vel]>0).if{~budget = ~budget - 1};
					[\budget, ~budget].postln;
					Routine{
						~event.(args[\inst], args[\pitch], args[\vel])
					}.play(SystemClock);
				}//{
					// query_new.();
			// };
			}{
				//player event is predicted
				// \ignored.postln;
				wait_time.max(0).sleep;
				query_new.();
				/*Routine{
					// TODO: this is ugly. should rethink globalArgs
					// to be overridden by explicit vals...
					var tmp_min_time = ~min_time;
					~min_time = wait_time;
					~noto.query(\handle, nil, *~noto.globalArgs);
					~min_time = tmp_min_time;
				}.play(SystemClock);*/
			};
			// return nil so the Routine is not rescheduled
            nil
        }.play(SystemClock);
    };
	(end==1 || (~step > 500)).if{
        //release the last note
        ~output.releaseAll(127);
		// ~noto.reset;
		~step = 0;
        \end.postln;
    };
};
)

(
~budget = 0;
~gate = false;
~last_event_time = nil;

// choose instruments for the player and Notochord
~player_inst = 272;
~player_drum = 300;
~exclude_inst = nil;
~noto_inst = ~output.instrumentMap.keys; //notochord instruments
// ~noto_inst = [7,8,16,25,26,33,34,35,36,46,105,106,107,108,130]; //notochord instruments
~allow_end = false;

~instrument_temp = 1;
~pitch_temp = 1; //0.95;
~rhythm_temp = 1;
~timing_temp = 0.2;
~min_time = 0.015;
~max_time = nil;
~velocity_temp = 0;
~include_pitch = nil;

~clear_prediction.();
~output.releaseAll;
~noto.reset;
// Routine{5.sleep; ~gate = true}.play;
)

~budget = -inf;
~gate = true;
Routine{~input.noteOnFn.(99, 35, 1); 0.2.sleep; ~input.noteOffFn.(0, 35, 1)}.play;
// ~noto.query(\handle, nil, *~noto.globalArgs);


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

