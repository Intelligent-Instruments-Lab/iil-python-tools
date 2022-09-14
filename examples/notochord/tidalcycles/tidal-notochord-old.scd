include("API");

(
thisProcess.openUDPPort(6789); // from tidal
~python = NetAddr.new("127.0.0.1", 9999); // to python
)

// notochord uses General MIDI (https://en.wikipedia.org/wiki/General_MIDI)
// instrument and drum mappings.
// instrument 1-128 are the GM melodic instruments, 129-256 are drumkits

// Option 1: map up to 16 notochord instruments to specific MIDI channels:
(
MIDIClient.init;
~m1 = MIDIOut.newByName("IAC Driver", "Bus 1").latency_(0);
// this will automatically restrict the instruments notochord can sample
// to the keys of ~midi_map
// (but make sure not to use any other instruments in tidal either)
// this example requires you to set up synths on MIDI channels 2,3,4, and 10
// where 10 has a drum rack.
// uncomment lines to add more channels, and change to first number to use
// different notochord instruments.
~midi_map = (
    // notochord instrument (1-256): midi channel (1-16)
    // 1: 1, //grand piano -> channel 1
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
// when using a drum instrument to the keys of ~drum_map
// (but make sure not to use any other drums in tidal either)
// this example is based on the default 808 rack in Ableton
~drum_map = (
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
//
/*~pitch_maps = (

)*/
~release_all = {arg vel=0;
    [~m1].do{arg port; 128.do{arg note; 16.do{arg chan;
        port.noteOff(chan, note, vel)}}}
};
~midi_send = {arg inst, pitch, vel;
    var channel = ~midi_map[inst]-1;
    ~is_drum.(inst).if{
        pitch = ~drum_map[pitch]
    };
    channel.isNil.if{"nil channel in ~midi_send: are you sending something from tidal which isn't in ~midi_map".postln;};
    pitch.isNil.if{"nil pitch in ~midi_send: are you sending something from tidal which isn't in ~drum_map?".postln;};
    (vel>0).if{
        ~m1.noteOn(channel, pitch, vel);
    }{
        ~m1.noteOff(channel, pitch);
    };
};
)

// Option 2: send MIDI to fluidsynth
// example fluidsynth terminal commands:
// fluidsynth -a coreaudio -m coremidi -v -o midi.autoconnect=1 -o synth.midi-bank-select=mma ~/Downloads/FluidR3_GM_GS.sf2
// fluidsynth -a coreaudio -m coremidi -v -o midi.autoconnect=1 -o synth.midi-bank-select=xg "/Users/victor/Downloads/Timbres of Heaven (XGM) 4.00(G)/Timbres of Heaven (XGM) 4.00(G).sf2"
(
~drum_map = nil;
~midi_map = nil;
~bank_xg = true;
~channel_lru = LinkedList.fill(16, {arg i; i});
~inst_channels = TwoWayIdentityDictionary.new;
~sema = Semaphore(1);

MIDIClient.init;
~m1 = MIDIOut.newByName("IAC Driver", "Bus 1").latency_(0);

~release_all = {arg vel=0;
    [~m1].do{arg port; 128.do{arg note; 16.do{arg chan;
        port.noteOff(chan, note, vel)}}}
};
~midi_send = {arg inst, pitch, vel;
    var channel;
    ~sema.wait;
    // check if this instrument has a channel
    channel = ~inst_channels.at(inst);
    channel.isNil.if{
        // if not get least recently used channel
        channel = ~channel_lru.popFirst;
        // and change the bank+program
        ~is_drum.(inst).if{
            ~m1.control(channel, 0, ~bank_xg.if{120}{1}); //drum
            ~m1.control(channel, 32, 0);
            ~m1.program(channel, inst-129); //program
        }{
            ~m1.control(channel, 0, 0); //melodic
            ~m1.control(channel, 32, 0);
            ~m1.program(channel, inst-1); //program
        };
        // TODO: anonymous instruments?

    }{
        ~channel_lru.remove(channel)
    };
    // send the event
    (vel>0).if{
        ~m1.noteOn(channel, pitch, vel);
    }{
        ~m1.noteOff(channel, pitch);
    };
    // update the inst / channel mappings
    ~channel_lru.add(channel);
    ~inst_channels.remove(channel);
    ~inst_channels[inst] = channel;
    ~sema.signal;
};
)


(
~oLatency = 0.1; // should match value at tidal boot
~ncLatency = ~oLatency-0.02; // slightly shorter than oLatency
~last_t = Process.elapsedTime;

~events = Dictionary.new;
~event_idx = 0;

// check if an instrument is a drumkit
~is_drum = {arg inst;
    ((inst>128)&&(inst<=256))||(inst>=265)
};

// handle OSC input from tidal
OSCdef(\from_tidal, {
    arg msg, time, src;
    var args = Dictionary.newFrom(msg[1..]);
    var delta = args[\delta] * (args[\legato]?1);
    // `time` is the start time
    // delta is the duration

    // \tidal_feed.postln;
    // time.postln;
    // [time - Process.elapsedTime].postln;
    // args.postln;

    ((args[\ncreset]?0) > 0).if{
        \reset.postln;
        SystemClock.schedAbs(time-~ncLatency, {
            ~python.sendMsg("/notochord/reset");
            ~release_all.(0);
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
            ~python.sendMsg(
                "/notochord/query_feed", \handle, handle,
                *~event_to_query.(event)
            )
        }{
            // else just feed notochord
            ~python.sendMsg(
                "/notochord/feed", *event.asPairs
            )
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
            ~midi_send.(event[\inst], event[\pitch], event[\vel]);
        };
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
            ~midi_send.(event[\inst], event[\pitch], 0);
        }
    });

};
// check if any sub-events not predicted yet
~any_missing = {arg event;
    event.collect(_<0).values.reduce('||')
};

// convert an event (constructed in schedule_event from data received from tidal)
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
OSCdef(\from_python, {
    arg msg, time, src;
    var args = Dictionary.newFrom(msg[1..]);
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


}, "notochord/query_return");

~python.sendMsg("/notochord/reset");
~release_all.(0);
)


(
~python.sendMsg("/notochord/reset");
~release_all.(0);
)
