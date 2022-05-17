(
thisProcess.openUDPPort(6789); // from tidal
~python = NetAddr.new("127.0.0.1", 9999); // to python
)


// for sending MIDI to ableton
(
MIDIClient.init;
// MIDIClient.destinations
~m1 = MIDIOut.newByName("IAC Driver", "Bus 1").latency_(0);
~m2 = MIDIOut.newByName("IAC Driver", "Bus 2").latency_(0);
~m3 = MIDIOut.newByName("IAC Driver", "Bus 3").latency_(0);
~prog2portchan = { arg prog;
    case
    {prog==0}{"can't convert start token".postln}
    {prog<=128}{
        var group = (prog-1 /8).asInteger;
        var idx = (prog-1 %8).asInteger;
        var port = switch(group)
        { 0}{case
            {idx<4}{~m1} //acoustic
            {idx<6}{~m2} //electric
            {true}{~m3} //harpsichord
        } //piano
        { 1}{((idx<3)||(idx==5)).if{~m1}{~m2}} //chromatic perc
        { 2}{(idx<4).if{~m1}{~m2}} //organ
        { 3}{(idx<5).if{~m1}{~m2}} //guitar
        { 4}{(idx<4).if{~m2}{~m1}} //bass
        { 5}{(idx<5).if{~m1}{~m2}} //strings
        { 6}{(idx<4).if{~m1}{~m2}} //ensemble
        { 7}{(idx<3).if{~m1}{~m2}} //brass
        { 8}{(idx<4).if{~m1}{~m2}} //reed
        { 9}{(idx<3).if{~m1}{~m2}} //pipe
        {10}{(idx%2==0).if{~m1}{~m2}} //synth lead
        {11}{(idx<4).if{~m1}{~m2}} //synth pad
        {12}{~m1} //synth fx
        {13}{case //'ethnic'
            {idx<=3}{~m1}
            {idx==4}{~m2}
            {idx==5}{group=2;~m2}
            {idx==6}{group=5;~m1}
            {idx==7}{group=8;~m2}
        }
        {14}{~m1} //percussive
        {15}{~m1} //sound fx
        ;
        // \melody.postln;
        (port:port, chan:group)
    }{(prog<=256)||(prog>=265)}{
        // \drum.postln;
        (port:~m2, chan:12)
    }{prog-257 < 8}{
        // \anon.postln;
        (port:~m1, chan:0)
    }
};
~release_all = {arg vel=0;
    [~m1, ~m2, ~m3].do{arg port; 128.do{arg note; 16.do{arg chan;
        port.noteOff(chan, note, vel)}}}
};
~midi_send = {arg inst, pitch, vel;
    var port_chan = ~prog2portchan.(inst);
    var port = port_chan[\port];
    var chan = port_chan[\chan];
    (vel>0).if{
        port.noteOn(chan, pitch, vel.max(1));
    }{
        port.noteOff(chan, pitch);
    }
}
)

OSCdef.trace(false);


// ok so how to actually 'schedule' things?
// SystemClock one-offs should do?
// schedule notochord-processing of each event for time of
//    oLatency before occurence
// also schedule the actual event to occur
// and set up a handle for the OSC return handler to
// fill in any predicted fields

(
~oLatency = 0.1; // must match value at tidal boot
~ncLatency = ~oLatency-0.02; // slightly shorter than oLatency
~last_t = Process.elapsedTime;

~events = Dictionary.new;
~event_idx = 0;

// handle OSC input from tidal
OSCdef(\from_tidal, {
    arg msg, time, src;
    var args = Dictionary.newFrom(msg[1..]);
    // `time` is the start time
    // args[\delta] is the duration
    \from_tidal.postln;
    time.postln;
    // [time - Process.elapsedTime].postln;
    args.postln;

    // note-on events should arrive from tidal in the order they happen
    // but note-offs do not, since they are implicit in duration of notes
    // the only guarantee is nothing needs to happen sooner than tidal's
    // oLatency parameter
    // so we can schedule all note-on and note-offs as soon as they arrive
    // (without all sub-events filled in, possibly)

    // anthing scheduled sooner than ncLatency (e.g. 100ms)
    // is 'frozen' and ready to (query_)feed notochord.

    // schedule note-on
    ~schedule_event.(
        time, args[\ncinst], args[\ncpitch], args[\ncvel]
    );
    // schedule note-off
    ~schedule_event.(
        time+args[\delta], args[\ncinst], args[\ncpitch], 0
    );

    // pop those off in order of occurence, and call feed or query_feed
    // in the case of query_feed, pass in some handle which tells the
    // return OSCdef where to put the result in the scheduler

    // finally pop completed events as they cross the current time,


    //when instrument, pitch or velocity is negative or nil,
    //query notochord

}, "/notochord/tidal_feed");

// create event handle,
// schedule processing by notochord,
// and schedule triggering the event
~schedule_event = {
    arg time, inst, pitch, vel;

    var handle = ~event_idx.asSymbol;
    var event = (inst:inst?(-1), pitch:pitch?(-1), vel:vel?(-1));
    ~events[handle] = event;
    ~event_idx = ~event_idx+1;
    ("scheduling event "++~event_idx).postln;

    // schedule notochord processing
    SystemClock.schedAbs(time-~ncLatency, {
        // compute dt from previous event
        event[\time] = time - ~last_t;
        \processing.postln; event.postln;
        ~last_t = time;
        (event[\time]<0).if{"warning: dt < 0".postln};

        // if any parts not determined
        (event.collect(_<0).values.reduce('||')).if{
            // query_feed
            // include handle to event for writing the result
            ~python.sendMsg(
                "/notochord/query_feed", \handle, handle,
                *~event_to_query.(event)
            )
        }{
            // else feed
            ~python.sendMsg(
                "/notochord/feed", *event.asPairs
            )
        }
    });

    // schedule event
    SystemClock.schedAbs(time, {
        // send to the synthesizer (MIDI bus or hacked dirth synth?)
        \playing.postln; event.postln;
    });

};

// convert an event (constructed in schedule_event from data received from tidal)
// to query keyword args for notochord
~event_to_query = {
    arg event;
    var query = List[];
    (event[\inst]>=0).if{query.add(\fix_instrument); query.add(event[\inst])};
    (event[\pitch]>=0).if{query.add(\fix_pitch); query.add(event[\pitch])};
    (event[\vel]>=0).if{query.add(\fix_vel); query.add(event[\vel])};
    query.add(\fix_time); query.add(event[\time]);
    query
};


// handle OSC return from python
OSCdef(\from_python, {
    arg msg, time, src;
    var args = Dictionary.newFrom(msg[1..]);
    var event = ~events[args[\handle]];

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

)
