Quarks.install("API")

// this the supercollider part of the tidal-notochord demo
// the tidal parts are in Notochord.hs and notochord.tidal

// currently supports sending MIDI to fluidsynth.

(
~noto = Notochord();

// ~noto.notochordCmd; // show the current notochord terminal command
~noto.notochordPath = "/Users/victor/notochord_lakh_50G_deep.pt";

~noto.startNotochord; // runs notochord for you in a terminal
)

( //////// OUTPUT OPTION A: fluidsynth ////////
~output = NotoFluidOutput();
// ~output.fluidSynthCmd //show the current fluidsynth terminal command
~output.soundFontPath = "/Users/victor/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2";
~output.startFluidSynth; // runs fluidsynth for you in a terminal
~allow_anon = false; // avoid anonymous instruments
)

( //////// OUTPUT OPTION B: MIDI mappings ////////
~output = NotoMappingOutput();
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
    // 257: 1 //first anonymous instrument -> channel 1
      8: 2, //clavinet -> channel 2
     13: 3, //marimba -> channel 3
     17: 4, //drawbar organ -> channel 4
     27: 5, //jazz guitar -> channel 5
    // 39: 6, //synth bass 1 -> channel 6
    // 47: 7, //harp -> channel 7
    // 54: 8, //voice oohs -> channel 8
    // 57: 9, //trumpet -> channel 9
    129:10, //standard drumkit -> channel 10 (drums)
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
thisProcess.openUDPPort(6789); // from tidal

~oLatency = 0.1; // should match value at tidal boot
~ncLatency = ~oLatency-0.02; // slightly shorter than oLatency
~last_t = Process.elapsedTime;

~events = Dictionary.new;
~event_idx = 0;

// handle OSC input from tidal
OSCdef(\from_tidal, {
    arg msg, time, src;
    var args = Dictionary.newFrom(msg[1..]);
    var delta = args[\delta] * (args[\legato]?1);
    // `time` is the start time
    // delta is the duration

    ((args[\ncreset]?0) > 0).if{
        \reset.postln;
        SystemClock.schedAbs(time-~ncLatency, {
            ~noto.reset;
            ~output.releaseAll;
        });
    }{
        // note-on events should arrive from tidal in the order they happen
        // but note-offs do not, since they are implicit in duration of notes.
        // the only guarantee about note-offs is they won't be scheduled sooner than tidal's
        // oLatency parameter, since that's the soonest a new note-on could occur.
        // so we can schedule all note-on and note-offs as soon as they arrive
        // (without all sub-events filled in, possibly)

        // anthing scheduled sooner than ncLatency (e.g. 100ms)
        // is 'frozen' and ready to (query_)feed notochord.

        // so we will schedule notochord processing to happen ~ncLatency
        // before the note-on,
        // ensuring each event gets updated before it is time to play it
        ~schedule_events.(
            time, delta, args[\ncinst], args[\ncpitch], args[\ncvel]
        );
    }

}, "/notochord/tidal_feed").permanent_(true);

// create event handle,
// schedule processing by notochord,
// and schedule triggering the note-on event and corresponding note-off
~schedule_events = {
    arg time, delta, inst, pitch, vel;

    var handle = ~event_idx.asSymbol;
    var event = (inst:inst?(-1), pitch:pitch?(-1), vel:vel?(-1));
    ~events[handle] = event;
    ~event_idx = ~event_idx+1;
    ("scheduling event "++handle).postln;

    // schedule notochord processing
    SystemClock.schedAbs(time - ~ncLatency, {
        // compute dt from previous event
        event[\time] = time - ~last_t;
        ("processing event "++handle).postln; event.postln;
        ~last_t = time;
        (event[\time]<0).if{"warning: dt < 0".postln};

        // if any parts not determined
        ~any_missing.(event).if{
            // query_feed notochord
            // include handle to event for writing the result
			~noto.queryFeed(\handle, handle, *~event_to_query.(event))
        }{
            // else just feed notochord
			~noto.feed(*event.asPairs)
        }
    });

    // schedule note-on
    // by the time this happens, the query_return OSCdef should have
    // completed the event (if it needed a query to notochord)
    SystemClock.schedAbs(time, {
        // send to the synthesizer (MIDI bus or hacked dirth synth?)
        ("starting note "++handle).postln; event.postln;
        ~any_missing.(event).if{
            "incomplete note-on".postln;
        }{
            ~output.send(event[\inst], event[\pitch], event[\vel]);
        }
    });

    // schedule note-off: fully determined once note-on is, since timing
    // comes from tidal. may revisit if allowing notochord to do micro-timing
    // in the future
    SystemClock.schedAbs(time+delta, {
        // send to the synthesizer (MIDI bus or hacked dirth synth?)
        ("ending note "++handle).postln; event.postln;
        ~any_missing.(event).if{
            "incomplete note-off".postln;
        }{
            ~output.send(event[\inst], event[\pitch], 0);
        }
    });

};
// check if any sub-events not predicted yet
~any_missing = {arg event;
    event.collect(_<0).values.reduce('||')
};

// convert an event (constructed in schedule_events from data received from tidal)
// to query keyword args for notochord
~event_to_query = {
    arg event;
    var query = List[];
    // if instrument is given
    (event[\inst]>=0).if{
		query.add(\next_inst); query.add(event[\inst])};
    // if pitch is given
    (event[\pitch]>=0).if{
		query.add(\next_pitch); query.add(event[\pitch])};
    // if velocity is given
    (event[\vel]>=0).if{
        query.add(\next_vel); query.add(event[\vel])
    }{
        // only predicting note-ons
        query.add(\min_vel); query.add(1)
    };
    query.add(\next_time); query.add(event[\time]);
	query.add(\allow_anon); query.add(~allow_anon);
	(~output.class==NotoMappingOutput).if{
        query.add(\include_inst);
        query.add(
			"%JSON:"++JSON.stringify(~output.instrumentMap.keys.asList));

		~output.drumMap.notNil.if{
			query.add(\include_drum);
			query.add(
				"%JSON:"++JSON.stringify(~output.drumMap.keys.asList))
		};
	};
	query
};

// handle OSC return from python
~noto.handler = { |args|
    var event = ~events[args[\handle]];

    ("updating event "++args[\handle]).postln;

    // update the event
    event[\inst] = args[\inst];
    event[\pitch] = args[\pitch];
    event[\vel] = args[\vel];

    ~events.removeAt(\handle);

    // TODO: spit a warning if trying to update
    // an event which should have happened already
    // (thisThread.seconds > ...).if{
    // "warning: updating past event".postln; event.postln};
};

~noto.reset;
~output.releaseAll;
)

// OSCFunc.trace(false)

// (
// ~noto.reset;
// ~output.releaseAll;
// )
