-- Notochord OSC target, OSC specs and Tidal parameters

-- See github.com/jarmitage/jarmlib for reference of 
-- how to incorporate this file into your Tidal setup

-- Notochord Target
:{
ncTarget = Target {
    oName      = "Notochord", 
    oAddress   = "127.0.0.1", 
    oHandshake = False, 
    oPort      = 6789, 
    oBusPort   = Nothing, 
    oLatency   = 0.1, 
    oSchedule  = Pre BundleStamp, 
    oWindow    = Nothing
    }
:}

-- need to use Named instead of ArgList to get
-- time from pattern structure (delta, cps, cycle)
:{
ncOSCSpecs = [
    OSC "/notochord/tidal_feed" $ Named {requiredArgs = []}
    ]
:}

-- control patterns
:{
let ncinst = pI "ncinst"
    ncpitch = pI "ncpitch"
    ncvel = pF "ncvel"
    ncreset = pI "ncreset"
    -- ncfeed i p t v = ncfeedI i # ncfeedP p # ncfeedT t # ncfeedV v
:}

-- Notochord OSC Map
ncOscMap = (ncTarget, ncOSCSpecs)
