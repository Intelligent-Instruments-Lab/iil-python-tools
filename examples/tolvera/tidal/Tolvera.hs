-- Lenia Target
:{
tolveraTarget = Target {oName     = "Tolvera",
                        oAddress   = "127.0.0.1",
                        oHandshake = True, 
                        oPort      = 7563, 
                        oBusPort   = Just 8001, 
                        oLatency   = 0.2,
                        oWindow    = Nothing,
                        oSchedule  = Pre BundleStamp
                       }
:}

-- Lenia OSC Specs
:{
tolveraOSCSpecs = [OSC "/lenia/world/{worldcmd}" $ ArgList [("worldslot", Nothing)],
                   OSC "/lenia/gui" $ ArgList [("guiupdaterate", Nothing)],
                   OSC "/lenia/pause" $ ArgList [("pause", Nothing)],
                   OSC "/lenia/reset" $ ArgList [("reset", Nothing)],
                   OSC "/lenia/time" $ ArgList [("time", Nothing)],
                   OSC "/lenia/convr" $ ArgList [("convr", Nothing)],
                   OSC "/lenia/grow/miu" $ ArgList [("growmiu", Nothing)],
                   OSC "/lenia/grow/sig" $ ArgList [("growsig", Nothing)],
                   OSC "/lenia/brush" $ ArgList [("brushmode", Nothing), -- none, draw, erase
                                                 ("brushradius", Nothing),
                                                 ("brushx", Nothing),
                                                 ("brushy", Nothing),
                                                 ("brushshow", Nothing)],
                   OSC "/boids/params" $ ArgList [("boidsseparate", Nothing),
                                                  ("boidsalign", Nothing),
                                                  ("boidscohere", Nothing),
                                                  ("boidsfear", Nothing),
                                                  ("boidsdt", Nothing),
                                                  ("boidsradius", Nothing),
                                                  ("boidsspeed", Nothing)],
                   OSC "/boids/pause" $ ArgList [("boidspause", Nothing)],
                   OSC "/boids/randomise" $ ArgList [("boidsrandomise", Nothing)],
                   OSC "/physarum/params" $ ArgList [("physarumsenseangle", Nothing),
                                                  ("physarumsensedist", Nothing),
                                                  ("physarumevaporation", Nothing),
                                                  ("physarummoveangle", Nothing),
                                                  ("physarummovestep", Nothing),
                                                  ("physarumsubstep", Nothing)]]
:}

-- Lenia Parameters
:{
let leniaworld       = pS "worldcmd"
    leniaworldslot   = pI "worldslot"
    leniapause       = pB "pause"
    leniaguiupdaterate = pF "guiupdaterate"
    leniareset       = pI "reset"
    leniatime        = pI "time"
    leniaconvr       = pI "convr"
    leniagrowmiu     = pF "growmiu"
    leniagrowsig     = pF "growsig"
    leniabrushmode   = pS "brushmode"
    leniabrushradius = pF "brushradius"
    leniabrushx      = pF "brushx"
    leniabrushy      = pF "brushy"
    leniabrushshow   = pB "brushshow"
:}

-- Boids Parameters
:{
let boidsseparate   = pF "boidsseparate"
    boidsalign      = pF "boidsalign"
    boidscohere     = pF "boidscohere"
    boidsfear       = pF "boidsfear"
    boidsdt         = pF "boidsdt"
    boidsradius     = pF "boidsradius"
    boidsspeed      = pF "boidsspeed"
    boidspause      = pS "boidspause" -- True/False
    _boidsrandomise = pI "boidsrandomise" -- True/False
    boidsrandomise  = _boidsrandomise 0
:}

-- Physarum Parameters
:{
let physarumsenseangle = pF "physarumsenseangle"
    physarumsensedist = pF "physarumsensedist"
    physarumevaporation = pF "physarumevaporation"
    physarummoveangle = pF "physarummoveangle"
    physarummovestep = pF "physarummovestep"
    physarumsubstep = pI "physarumsubstep"
:}

-- OSC Map
tolveraOscMap = (tolveraTarget, tolveraOSCSpecs)
