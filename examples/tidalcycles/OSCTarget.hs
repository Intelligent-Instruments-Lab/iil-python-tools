-- Target
:{
iipyTarget = Target {oName     = "iipyper",
                   oAddress   = "127.0.0.1",
                   oHandshake = True, 
                   oPort      = 8000, 
                   oBusPort   = Just 8001, 
                   oLatency   = 0.2,
                   oWindow    = Nothing,
                   oSchedule  = Pre BundleStamp
                  }
:}

-- OSC Specs
:{
iipyOSCSpecs = [OSC "/path/{path1}/" $ ArgList [("param1", Nothing),
                                                ("delta",  Just $ VF 0),
                                                ("cycle",  Just $ VF 0),
                                                ("cps",    Just $ VF 0)],
                OSC "/path/{path2}/" $ ArgList [("param2", Nothing),
                                                ("delta",  Just $ VF 0),
                                                ("cycle",  Just $ VF 0),
                                                ("cps",    Just $ VF 0)]]
:}

-- Parameters
:{
let path1  = pS "path1"
    path2  = pS "path2"
    param1 = pF "param1"
    param2 = pF "param2"
:}

-- OSC Map
iipyOscMap = (iipyTarget, iipyOSCSpecs)
