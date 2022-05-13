{-

Notochord OSC target, OSC specs and Tidal parameters

See github.com/jarmitage/jarmlib for reference of 
how to incorporate this file into your Tidal setup

-}

-- Notochord Target
:{
ncTarget = Target {oName     = "Notochord", 
                   oAddress   = "127.0.0.1", 
                   oHandshake = False, 
                   oPort      = 6789, 
                   oBusPort   = Nothing, 
                   oLatency   = 0.1, 
                   oSchedule  = Pre BundleStamp, 
                   oWindow    = Nothing
                  }
:}

{-
Notochord.feed() consumes an event and advance hidden state
    
inst: int. instrument of current note.
    0 is start token
    1-128 are General MIDI instruments
    129-256 are drumkits (MIDI 1-128 on channel 13)
    257-264 are 'anonymous' melodic instruments
    265-272 are 'anonymous' drumkits
pitch: int. MIDI pitch of current note.
    0-127 are MIDI pitches / drums
    128 is start token
time: float. elapsed time in seconds since previous event.
vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
    0 indicates a note-off event
-}

-- Notochord Feed OSC Specs
:{
ncFeedOSCSpecs = [OSC "/notochord/feed/inst"  $ ArgList [("ncfeedI", Nothing)],
                  OSC "/notochord/feed/pitch" $ ArgList [("ncfeedP", Nothing)],
                  OSC "/notochord/feed/time"  $ ArgList [("ncfeedT", Nothing)],
                  OSC "/notochord/feed/vel"   $ ArgList [("ncfeedV", Nothing)]]
:}

-- Notochord Feed Parameters
:{
let ncfeedI = pI "ncfeedI"
    ncfeedP = pI "ncfeedP"
    ncfeedT = pF "ncfeedT"
    ncfeedV = pF "ncfeedV"
    ncfeed i p t v = ncfeedI i # ncfeedP p # ncfeedT t # ncfeedV v
:}

{-
Notochord.query() returns a prediction for the next note.
various constraints on the the next note can be requested.

# hard constraints
fix_*: same as the arguments to feed, but to fix a value for the predicted note.
    sampled values will always condition on fixed values, so passing
    `fix_instrument=1`, for example, will make the event appropriate
    for the piano (instrument 1) to play.
    
# partial constraints
allow_end: if False, zero probability of sampling the end marker
min_time, max_time: if not None, truncate the time distribution
include_instrument: instrument id(s) to include in sampling.
    (if not None, all others will be excluded)
exclude_instrument: instrument id(s) to exclude from sampling.
include_pitch: pitch(es) to include in sampling.
    (if not None, all others will be excluded)
exclude_pitch: pitch(es) to exclude from sampling.
min_vel, max_vel: if not None, truncate the velocity distribution

# sampling strategies
instrument_temp: if not None, apply top_p sampling to instrument. 0 is
    deterministic, 1 is 'natural' according to the model
pitch_temp: if not None, apply top_p sampling to pitch. 0 is
    deterministic, 1 is 'natural' according to the model
velocity_temp: if not None, apply temperature sampling to the velocity
    component.
rhythm_temp: if not None, apply top_p sampling to the weighting
    of mixture components. this affects coarse rhythmic patterns;
    0 is deterministic, 1 is 'natural' according to the model
timing_temp: if not None, apply temperature sampling to the time
    component. this affects fine timing; 0 is deterministic and 
    precise, 1 is 'natural' according to the model.
index_pitch: Optional[int]. if not None, deterministically take the
    nth most likely pitch instead of sampling.

# multiple predictions
pitch_topk: Optional[int]. if not None, instead of sampling pitch, 
    stack the top k most likely pitches along the batch dimension
sweep_time: if True, instead of sampling time, choose a diverse set of
    times and stack along the batch dimension

Returns: dict of
'end': int. value of 1 indicates the *current* event (the one 
    passed as arguments to `predict`) was the last event, and the
    predicted event should *not* be played. if `allow end` is false, 
    this will always be 0.
'step': int. number of steps since calling `reset`.
'instrument': int. id of predicted instrument.
    1-128 are General MIDI standard melodic instruments
    129-256 are drumkits for MIDI programs 1-128
    257-264 are 'anonymous' melodic instruments
    265-272 are 'anonymous' drums
'pitch': int. predicted MIDI number of next note, 0-128.
'time': float. predicted time to next note in seconds.
'velocity': float. unquantized predicted velocity of next note.
    0-127; hard 0 indicates a note-off event.
'*_params': tensor. distribution parameters for visualization
    purposes.
-}

-- Notochord Query OSC Specs
:{
ncQueryOSCSpecs = [OSC "/notochord/query/fix/inst"  $ ArgList [("ncqueryfixI", Nothing)],
                   OSC "/notochord/query/fix/pitch" $ ArgList [("ncqueryfixP", Nothing)],
                   OSC "/notochord/query/fix/time"  $ ArgList [("ncqueryfixT", Nothing)],
                   OSC "/notochord/query/fix/vel"   $ ArgList [("ncqueryfixV", Nothing)],
                   OSC "/notochord/query/constrain/excludei" $ ArgList [("ncqueryexcludeI", Nothing)],
                   OSC "/notochord/query/constrain/includep" $ ArgList [("ncqueryincludeP", Nothing)],
                   OSC "/notochord/query/constrain/excludep" $ ArgList [("ncqueryexcludeP", Nothing)],
                   OSC "/notochord/query/constrain/mint"     $ ArgList [("ncqueryminT", Nothing)],
                   OSC "/notochord/query/constrain/maxt"     $ ArgList [("ncquerymaxT", Nothing)],
                   OSC "/notochord/query/constrain/minv"     $ ArgList [("ncqueryminV", Nothing)],
                   OSC "/notochord/query/constrain/maxv"     $ ArgList [("ncquerymaxV", Nothing)],
                   OSC "/notochord/query/sample/ncquerytempi"  $ ArgList [("ncquerytempI", Nothing)],
                   OSC "/notochord/query/sample/ncquerytempp"  $ ArgList [("ncquerytempP", Nothing)],
                   OSC "/notochord/query/sample/ncquerytempt"  $ ArgList [("ncquerytempT", Nothing)],
                   OSC "/notochord/query/sample/ncquerytempr"  $ ArgList [("ncquerytempR", Nothing)],
                   OSC "/notochord/query/sample/ncquerytempv"  $ ArgList [("ncquerytempV", Nothing)],
                   OSC "/notochord/query/sample/ncqueryindexp" $ ArgList [("ncqueryindexP", Nothing)],
                   OSC "/notochord/query/multi/ncquerytopp"   $ ArgList [("ncquerytopP", Nothing)],
                   OSC "/notochord/query/multi/ncquerysweept" $ ArgList [("ncquerysweepT", Nothing)]
                  ]
:}

-- Notochord Query: fix
:{
let ncqueryfixI = pI "ncqueryfixI"
    ncqueryfixP = pI "ncqueryfixP"
    ncqueryfixT = pF "ncqueryfixT"
    ncqueryfixV = pF "ncqueryfixV"
:}

-- Notochord Query: constraints
:{
let ncqueryexcludeI = pI "ncqueryexcludeI"
    ncqueryincludeP = pI "ncqueryincludeP"
    ncqueryexcludeP = pI "ncqueryexcludeP"
    ncqueryminT = pF "ncqueryminT"
    ncquerymaxT = pF "ncquerymaxT"
    ncqueryminV = pF "ncqueryminV"
    ncquerymaxV = pF "ncquerymaxV"
:}

-- Notochord Query: sampling
:{
let ncquerytempI  = pF "ncquerytempI"
    ncquerytempP  = pF "ncquerytempP"
    ncquerytempT  = pF "ncquerytempT"
    ncquerytempR  = pF "ncquerytempR"
    ncquerytempV  = pF "ncquerytempV"
    ncqueryindexP = pI "ncqueryindexP"
:}

-- Notochord Query: multiple predictions
:{
let ncquerytopP   = pI "ncquerytopP"
    ncquerysweepT = pI "ncquerysweepT"
:}

let ncOSCSpecs = ncFeedOSCSpecs++ncQueryOSCSpecs

-- Notochord OSC Map
ncOscMap = (ncTarget, ncOSCSpecs)
