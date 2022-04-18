(
Server.default.options.numInputBusChannels = 8;
s.boot;
)

(
SynthDef(\analyzer, {
    arg in=0, threshold=0.1, relaxtime=0.05;
    var onset1, s1, lv1;
    var sdel = 0.04; // this is a small delay to skip the first chaotic miliseconds

    var sound_in = HPF.ar(SoundIn.ar(in), 30);

    var amp = WAmp.kr(sound_in, sdel);

    var f0 = Pitch.kr(sound_in, 300,
        ampThreshold:0.001, peakThreshold:0.5, minFreq:30, maxFreq:8000, median:5
    )[0];

    var onsets = Onsets.kr(FFT(LocalBuf(2048), sound_in, wintype:1), threshold, \wphase,//\rcomplex,
        relaxtime:relaxtime, floor:0.1, mingap:1, medianspan:11, whtype:1, rawodf:0);

    var delay_onsets = DelayN.kr(onsets, sdel, sdel);

    // onsets.poll(5);

    SendReply.kr(delay_onsets, '/analysis', [amp, f0]);
}).add;
)

(
~analyzer!?(_.free);
~analyzer = Synth(\analyzer, [\in, 6, \threshold, 0.1]);
)

(
~clear_pending_notes.();
b.sendMsg("/predictor/reset");
~release_all.(0);
~allow_end = false;
~duet = true;
~do_synth = false;
~player_sound = true;
~player_inst = 43;
~last_pitch = nil;
~gate = false;

OSCdef(\analysis, {
    arg msg;
    var amp = msg[3];
    var f0 = msg[4];
    var vel = (amp.pow(0.5)*126+1).round.asInteger;
    var pitch = f0.cpsmidi.round.asInteger;


    ~last_pitch.notNil.if{
        MIDIdef.all[\input_off].func.value(0, ~last_pitch)};
    MIDIdef.all[\input_on].func.value(vel, pitch);

    ~last_pitch = pitch;

}, "/analysis")
)

(
~clear_pending_notes.();
~release_all.(0);
)


// ~analyzer.set(\threshold, 0.1)

// OSCdef.trace
