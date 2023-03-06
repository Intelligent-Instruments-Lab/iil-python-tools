-- Lenia Target
:{
leniaTarget = Target {oName     = "Lenia",
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
leniaOSCSpecs = [OSC "/lenia/world/{worldcmd}" $ ArgList [("worldslot", Nothing)],
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
                                               ("brushshow", Nothing)]]
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

-- OSC Map
leniaOscMap = (leniaTarget, leniaOSCSpecs)
