-- https://github.com/musikinformatik/SuperDirt/issues/276

-- TODO
-- [ ] `offsets` doesnt work for pattern with one value..?
-- [ ] Swap `mrpnote` for `n`, but find a way to round it. roundn _n = mapM round (n _n)
-- [ ] Refine mrp_n shortcut to have multiple args
-- [ ] Redo harmonics to have a map / use arrays
-- [ ] add hush etc to lib

import Data.Maybe (fromJust)

:{
eventHasOffset :: Event a -> Bool
eventHasOffset e | isAnalog e = False
                 | otherwise = stop (fromJust $ whole e) == stop (part e)
:}

:{
offsets :: Pattern a -> Pattern a
offsets pat = withEvent f $ filterEvents eventHasOffset pat
  where f (e@(Event _ Nothing _ _)) = e -- ignore analog events
        f (Event c (Just (Arc b e)) _ v) = Event c (Just a) a v
           where a = Arc e (e + (e - b))
:}

-- MRP Target
:{
mrpTarget = Target {oName     = "MRP",
                   oAddress   = "127.0.0.1",
                   oHandshake = False, 
                   oPort      = 7770, 
                   oBusPort   = Just 8001, 
                   oLatency   = 0.2,
                   oWindow    = Nothing,
                   oSchedule  = Pre BundleStamp
                  }
:}

-- MRP OSC Specs
:{
mrpOSCSpecs = [OSC "/mrp/midi" $ ArgList [("mrpstatus", Nothing),
                                          ("mrpnote", Nothing),
                                          ("mrpvelocity", Just $ VI 1)],
              OSC "/mrp/quality/intensity" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrpintensitynote", Nothing),
                                          ("mrpintensity", Nothing)],
              OSC "/mrp/quality/brightness" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrpbrightnessnote", Nothing),
                                          ("mrpbrightness", Nothing)],
              OSC "/mrp/quality/pitch" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrppitchnote", Nothing),
                                          ("mrppitch", Nothing)],
              OSC "/mrp/quality/pitch/vibrato" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrppitchvibratonote", Nothing),
                                          ("mrppitchvibrato", Nothing)],
              OSC "/mrp/quality/harmonic" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrpharmonicnote", Nothing),
                                          ("mrpharmonic", Nothing)],
              OSC "/mrp/quality/harmonics/raw" $ ArgList [("mrpchannel", Just $ VI 15),
                                          ("mrpharmonicsnote", Nothing),
                                          ("mrpharmonics0", Just $ VF 0),
                                          ("mrpharmonics1", Just $ VF 0),
                                          ("mrpharmonics2", Just $ VF 0),
                                          ("mrpharmonics3", Just $ VF 0),
                                          ("mrpharmonics4", Just $ VF 0),
                                          ("mrpharmonics5", Just $ VF 0),
                                          ("mrpharmonics6", Just $ VF 0),
                                          ("mrpharmonics7", Just $ VF 0),
                                          ("mrpharmonics8", Just $ VF 0),
                                          ("mrpharmonics9", Just $ VF 0),
                                          ("mrpharmonics10", Just $ VF 0),
                                          ("mrpharmonics11", Just $ VF 0),
                                          ("mrpharmonics12", Just $ VF 0),
                                          ("mrpharmonics13", Just $ VF 0),
                                          ("mrpharmonics14", Just $ VF 0),
                                          ("mrpharmonics15", Just $ VF 0),
                                          ("mrpharmonics16", Just $ VF 0)],
              OSC "/mrp/allnotesoff" $ ArgList [("mrphush", Nothing)]]
:}

-- MRP MIDI Note
:{
let mrp_status = pI "mrpstatus"
    mrp_note = pI "mrpnote"
    mrp_velocity = pI "mrpvelocity"
    mrp_n _n = stack [
      mrp_note _n # mrp_velocity 1 # mrp_status 0x9F,
      offsets (mrp_note _n # mrp_velocity 0) # mrp_status 0x8F]
:}

-- MRP Sound Qualities
:{
let _mrp_intensity      = pF "mrpintensity"
    _mrp_brightness     = pF "mrpbrightness"
    _mrp_pitch          = pF "mrppitch"
    _mrp_pitch_vibrato  = pF "mrppitchvibrato"
    _mrp_harmonic_sweep = pF "mrpharmonic"
    _mrp_harmonics      = pF "mrpharmonic"
    mrp_intensity_note      = pI "mrpintensitynote"
    mrp_brightness_note     = pI "mrpbrightnessnote"
    mrp_pitch_note          = pI "mrppitchnote"
    mrp_pitch_vibrato_note  = pI "mrppitchvibratonote"
    mrp_harmonic_sweep_note = pI "mrpharmonicnote"
    mrp_harmonics_note      = pI "mrpharmonicsnote"
    mrp_intensity _n _v      = _mrp_intensity _v + mrp_intensity_note _n
    mrp_brightness _n _v     = _mrp_brightness _v + mrp_brightness_note _n
    mrp_pitch _n _v          = _mrp_pitch _v + mrp_pitch_note _n
    mrp_pitch_vibrato _n _v  = _mrp_pitch_vibrato _v + mrp_pitch_vibrato_note _n
    mrp_harmonic_sweep _n _v = _mrp_harmonic_sweep _v + mrp_harmonic_sweep_note _n
    mrp_harmonics _n _v      = _mrp_harmonics _v + mrp_harmonics_note _n
:}

-- MRP Sound Qualities - Harmonics
:{
let _mrp_h0 = pF "mrpharmonics0"
    _mrp_h1 = pF "mrpharmonics1"
    _mrp_h2 = pF "mrpharmonics2"
    _mrp_h3 = pF "mrpharmonics3"
    _mrp_h4 = pF "mrpharmonics4"
    _mrp_h5 = pF "mrpharmonics5"
    _mrp_h6 = pF "mrpharmonics6"
    _mrp_h7 = pF "mrpharmonics7"
    _mrp_h8 = pF "mrpharmonics8"
    _mrp_h9 = pF "mrpharmonics9"
    _mrp_h10 = pF "mrpharmonics10"
    _mrp_h11 = pF "mrpharmonics11"
    _mrp_h12 = pF "mrpharmonics12"
    _mrp_h13 = pF "mrpharmonics13"
    _mrp_h14 = pF "mrpharmonics14"
    _mrp_h15 = pF "mrpharmonics15"
    _mrp_h16 = pF "mrpharmonics16"
    mrp_h0 _n _h = _mrp_h0 _h # mrp_harmonics_note _n
    mrp_h1 _n _h = _mrp_h1 _h # mrp_harmonics_note _n
    mrp_h2 _n _h = _mrp_h2 _h # mrp_harmonics_note _n
    mrp_h3 _n _h = _mrp_h3 _h # mrp_harmonics_note _n
    mrp_h4 _n _h = _mrp_h4 _h # mrp_harmonics_note _n
    mrp_h5 _n _h = _mrp_h5 _h # mrp_harmonics_note _n
    mrp_h6 _n _h = _mrp_h6 _h # mrp_harmonics_note _n
    mrp_h7 _n _h = _mrp_h7 _h # mrp_harmonics_note _n
    mrp_h8 _n _h = _mrp_h8 _h # mrp_harmonics_note _n
    mrp_h9 _n _h = _mrp_h9 _h # mrp_harmonics_note _n
    mrp_h10 _n _h = _mrp_h10 _h # mrp_harmonics_note _n
    mrp_h11 _n _h = _mrp_h11 _h # mrp_harmonics_note _n
    mrp_h12 _n _h = _mrp_h12 _h # mrp_harmonics_note _n
    mrp_h13 _n _h = _mrp_h13 _h # mrp_harmonics_note _n
    mrp_h14 _n _h = _mrp_h14 _h # mrp_harmonics_note _n
    mrp_h15 _n _h = _mrp_h15 _h # mrp_harmonics_note _n
    mrp_h16 _n _h = _mrp_h16 _h # mrp_harmonics_note _n
    mrp_h _n h = _mrp_h0 (h!!0) # _mrp_h1 (h!!1) # _mrp_h2 (h!!2) # _mrp_h3 (h!!3) # _mrp_h4 (h!!4) # _mrp_h5 (h!!5) # _mrp_h6 (h!!6) # _mrp_h7 (h!!7) # _mrp_h8 (h!!8) # _mrp_h9 (h!!9) # _mrp_h10 (h!!10) # _mrp_h11 (h!!11) # _mrp_h12 (h!!12) # _mrp_h13 (h!!13) # _mrp_h14 (h!!14) # _mrp_h15 (h!!15) # _mrp_h16 (h!!16) # mrp_harmonics_note _n
    -- mrp_hi _n _i _h = # mrp_harmonics_note _n -- TODO
    -- mrp_his _n _i _h = heap / stack version
:}

-- MRP Hush etc
:{
let _mrp_hush = pI "mrphush"
    mrp_hush = _mrp_hush 1
    -- hushmrp = do
    --   hush
    --   once $ mrp_hush
:}

-- MRP Parameter shortcuts/groups
:{
let mrp_n' _n _i _b _p = stack [
      mrp_note _n # mrp_velocity 127 # mrp_status 0x9F,
      offsets (mrp_note _n # mrp_velocity 0) # mrp_status 0x8F,
      mrp_intensity _n _i,
      mrp_brightness _n _b,
      mrp_pitch _n _p]
:}

-- MRP OSC Map
mrpOscMap = (mrpTarget, mrpOSCSpecs)
