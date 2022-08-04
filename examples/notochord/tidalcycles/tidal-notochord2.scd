Quarks.install("API")

(
~noto = Notochord();

// ~noto.notochordCmd; // show the current notochord terminal command
~noto.notochordPath = "/Users/victor/Downloads/notochord_lakh_20G.ckpt";

~noto.startNotochord; // runs notochord for you in a terminal
)

(
~output = NotoFluidOutput();

// ~output.fluidSynthCmd //show the current fluidsynth terminal command
~output.soundFontPath = "/Users/victor/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2";

~output.startFluidSynth; // runs fluidsynth for you in a terminal
)

// notochord uses General MIDI (https://en.wikipedia.org/wiki/General_MIDI)
// instrument and drum mappings.
// instrument 1-128 are the GM melodic instruments, 129-256 are drumkits

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
            ~output.resetAll;
        });
    }{
        // note-on events should arrive from tidal in the order they happen
        // but note-offs do not, since they are implicit in duration of notes.
        // the only guarantee is they won't need to happen sooner than tidal's
        // oLatency parameter, since that's the soonest a new note-on can occur.
        // so we can schedule all note-on and note-offs as soon as they arrive
        // (without all sub-events filled in, possibly)

        // anthing scheduled sooner than ncLatency (e.g. 100ms)
        // is 'frozen' and ready to (query_)feed notochord.

        // so we will schedule notochord processing to happen ~ncLatency
        // before the note-on,
        // ensuring the event gets updated before it is time to play it
        ~schedule_events.(
            time, delta, args[\ncinst], args[\ncpitch], args[\ncvel]
        );
    }

}, "/notochord/tidal_feed");

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
    (event[\inst]>=0).if{query.add(\fix_instrument); query.add(event[\inst])};
    // if pitch is given
    (event[\pitch]>=0).if{query.add(\fix_pitch); query.add(event[\pitch])};
    // if velocity is given
    (event[\vel]>=0).if{
        query.add(\fix_vel); query.add(event[\vel])
    }{
        // only predicting note-ons
        query.add(\min_vel); query.add(1)
    };
    query.add(\fix_time); query.add(event[\time]);
    ~midi_map.notNil.if{
        query.add(\include_instrument);
        query.add("%JSON:"++JSON.stringify(~midi_map.keys.asList))
    };
    ~drum_map.notNil.if{
        query.add(\include_drum);
        query.add("%JSON:"++JSON.stringify(~drum_map.keys.asList))
    };
    query
};

// handle OSC return from python
~noto.handler = { |args|
    var event = ~events[args[\handle]];

    ("updating event "++args[\handle]).postln;

    // update the event
    event[\inst] = args[\instrument];
    event[\pitch] = args[\pitch];
    event[\vel] = args[\velocity];

    ~events.removeAt(\handle);

    // TODO: spit a warning if it trying to update
    //  an event which should have happened already)
    // (thisThread.seconds > ...).if{
    // "warning: updating past event".postln; event.postln};
};

~noto.reset;
~output.releaseAll;
)

OSCFunc.trace(false)

(
~noto.reset;
~output.releaseAll;
)
