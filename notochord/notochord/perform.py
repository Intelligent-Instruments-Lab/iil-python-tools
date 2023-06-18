from typing import Optional, Dict, List, Tuple, Any
import time

import pandas as pd
import numpy as np

class MIDIConfig(dict):
    """
    invertible map from MIDI channel: Notochord instrument
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.invertible = len(self.channels)==len(self.insts)

    @property
    def channels(self):
        """set of channels"""
        return set(self)
    @property
    def insts(self):
        """set of instruments"""
        return set(self.values())
    def inv(self, inst):
        """map from Notochord instrument: MIDI channel"""
        if not self.invertible:
            print('WARNING: MIDIConfig is not invertible')
        for chan,inst_ in self.items():
            if inst_==inst:
                return chan
        raise KeyError(f"""
            instrument {inst} has no channel
            """)
    
class NotoPerformance:
    """
    track various quantities of a Notochord performance:

    event history:
        * wall time
        * nominal dt
        * pitch
        * velocity (0 for noteoff)
        * notochord instrument
    
    query for:
        * instruments present in the last N events
        * number of note_ons by instrument in last N events
        * currently playing notes with user data as {(inst, pitch): Any}
        * currently playing notes as {inst: pitches}
    """
    def __init__(self):
        self.init()
        self.notes:Dict[Tuple[int,int], Any] = {} 
        self.past_segments:List[pd.DataFrame] = []

    def init(self):
        self.events = pd.DataFrame(np.array([],dtype=[
            ('wall_time_ns',np.int64), # actual wall time played in ns
            ('time',np.float32), # nominal notochord dt in seconds
            ('inst',np.int16), # notochord instrument
            ('pitch',np.int16), # MIDI pitch
            ('vel',np.int8), # MIDI velocity
            ('channel',np.int8), # MIDI channel
            ]))
        
    def push(self):
        """push current events onto a list of past segments,
            start a fresh history
        """
        self.past_segments.append(self.events)
        self.init()
        
    def feed(self, held_note_data:Any=None, **event):
        """
        Args:
            held_note_data: any Python object to be attached to held notes
                (ignored for note-offs)
            ('wall_time_ns',np.int64), # actual wall time played in ns
            ('time',np.float32), # nominal notochord dt in seconds
            ('inst',np.int16), # notochord instrument
            ('pitch',np.int16), # MIDI pitch
            ('vel',np.int8), # MIDI velocity
            ('channel',np.int8), # MIDI channel
        """
        if 'wall_time_ns' not in event:
            event['wall_time_ns'] = time.time_ns()
        if 'channel' not in event:
            # use -1 for missing channel to avoid coercion to float
            event['channel'] = -1 
        cast_event = {}
        for k,v in event.items():
            if k in self.events.columns:
                cast_event[k] = self.events.dtypes[k].type(v)
        event = cast_event

        self.events.loc[len(self.events)] = event

        chan = event.get('channel', None)
        inst, pitch, vel = event['inst'], event['pitch'], event['vel']
        k = (chan, inst, pitch)
        if vel > 0:
            self.notes[k] = held_note_data
        else:
            self.notes.pop(k, None)
   
    def inst_counts(self, n=0, insts=None):
        """instrument counts in last n (default all) note_ons"""
        df = self.events
        df = df.loc[df.vel > 0]
        df = df.iloc[-n:]
        counts = df.inst.value_counts()
        for inst in insts:
            if inst not in counts.index:
                counts[inst] = 0
        return counts
    
    def held_inst_pitch_map(self, insts=None):
        """held notes as {inst:[pitch]} for given instruments"""
        if insts is None:
            insts = self.events.inst.unique()
        note_map = {i:[] for i in insts}
        for c,i,p in self.notes:
            if i in insts:
                note_map[i].append(p)
        return note_map
    
    @property
    def note_pairs(self):
        """held notes as {(inst,pitch)}.
        returns a new `set`; safe to modify history while iterating
        """
        return {(i,p) for (c,i,p) in self.notes}
    
    @property
    def note_triples(self):
        """held notes as {(channel,inst,pitch)}.
        returns a new `set`; safe to modify history while iterating
        """
        return set(self.notes)

    @property
    def note_data(self):
        """held notes as {(chan,inst,pitch):held_note_data}.
        mutable.
        """
        # NOTE: returned dictionary should be mutable
        return self.notes


class KlaisOrganManual:
    def __init__(self, name, channel, note_range, 
                 voices, mixtures, tremulant, couplings):
        self.name = name,
        self.channel = channel # 1-indexed
        self.note_range = range(note_range[0], note_range[1]+1)
        self.voices = voices # {transpose: {note: name}}
        self.mixtures = mixtures
        self.termulant:Optional[int] = tremulant # MIDI note or None
        self.coupling = couplings

class KlaisOrganConfig:
    def __init__(self):
        self.effects = {
            87:'Cymbelstern A',
            88:'Cymbelstern B',
            89:'Nachtigall'
        }
        self.manuals = [
            KlaisOrganManual(
                name = 'Bombardewerk / Solo',
                channel = 5, # 1-indexed
                note_range = (36,93), # inclusive # (c-a''')
                voices = {
                    0:{# MIDI transpose
                        0:"Rohrflöte 8'",#MIDI note: name
                        4:"Chamade 8'",
                        6:"Orlos 8'	",
                    },
                    12:{
                        1:"Praestant 4'",
                        5:"Chamade 4'",
                    },
                    -12:{
                        3:"Chamade 16'",
                    }
                },
                mixtures = {
                    2:"Cornet 3f",#  (Not full range of keyboard, g-a''')
                },
                tremulant = 7,
                couplings = {
                    8:'Super III an II',
                    9:'Sub III an II'
                }
            ),
            KlaisOrganManual(
                name = 'Scwellwerk / Swell',
                channel = 4, # 1-indexed
                note_range = (36,93), # inclusive # (c-a''')
                voices = {
                    0:{
                        10:" Salicet 8'",
                        11:"Geigenprincipal 8'",
                        12:"Flute harm. 8'",
                        13:"Bourdon 8'",
                        14:"Gamba 8'",
                        15:"Vox coelestis 8'",#	 (c-a''')
                        25:"Trom. Harm. 8'",
                        26:"Hautbois 8'",
                        27:"Vox humana 8'",
                    },
                    12:{
                        16:"Octave 4'",
                        17:"Flute octav. 4'",
                        18:"Salicional 4'",
                        28:"Clairon harm. 4'",        
                    },
                    24:{
                        19:"Octavin 2'",
                    },
                    36:{
                        20:"Piccolo 1'",
                    },
                    19:{
                        21:"Nasard 2 2/3'",
                    },
                    28:{
                        22:"Terz 1 3/5'",
                    },
                    -12:{
                        24:"Basson 16'",
                    }
                },
                mixtures = {
                    23:"Fourniture 6f",
                },
                tremulant = 29,
                couplings = {
                    30:'IV an III',
                }
            ),
            KlaisOrganManual(
                name = 'Hauptwerk / Great',
                channel = 3, # 1-indexed
                note_range = (36,93), # inclusive # (c-a''')
                voices = {
                    0:{
                        33:"Principal 8'",
                        34:"Doppelflöte 8'",
                        35:"Gemshorn 8'",
                        46:"Trompete 8'",    
                    },
                    12:{
                        36:"Octave 4'",
                        37:"Nachthorn 4'",
                        47:"Trompete 4'",
                    },
                    -12:{
                        31:"Praestant 16'",
                        32:"Bourdon 16'",
                        45:"Trompete 16'",                    
                    },
                    24:{
                        38:"Superoctave 2'",
                    },
                    7:{
                        39:"Quinte 5 1/3'",
                    },
                    16:{
                        40:"Terz 3 1/5'",
                    },
                    19:{
                        41:"Quinte 2 2/3'",
                    }
                },
                mixtures = {
                    42:"Cornet 5f", # (from c’)
                    43:"Mixtur 5f",
                    44:"Acuta 4f",
                },
                tremulant = None,
                couplings = {
                    48:'I an II',
                    49:'III an I',
                    50:'IV an II',
                }
            ),
            KlaisOrganManual(
                name = 'Rückpositiv / Positive',
                channel = 2, # 1-indexed
                note_range = (36,93), # inclusive # (c-a''')
                voices = {
                    0:{
                        51:"Praestant 8'",
                        52:"Gedackt 8'",
                        53:"Quintade 8'",
                        63:"Trompete 8'",
                        64:"Cromorne 8'",        
                    },
                    12:{
                        54:"Principal 4'",
                        55:"Rohrflöte 4'",
                    },
                    24:{
                        56:"Octave 2'",
                        57:"Waldflöte 2'",
                    },
                    31:{
                        58:"Larigot 1 1/3'",
                    },
                    -12:{
                        62:"Dulcian 16'",
                    }
                },
                mixtures = {
                    59:"Sesquialter 2f",
                    60:"Scharff 5f",
                    61:"Cymbel 4f",
                },
                tremulant = 65,
                couplings = {
                    66:'II an III',
                }
            ),
            KlaisOrganManual(
                name = 'Pedal / Pedalboard',
                channel = 1, # 1-indexed
                note_range = (36,67), # inclusive # (c-g')
                voices = {
                    0:{
                        69:"Oktave 8'",
                        75:"Posaune 8'",
                        80:"Spielflöte 8'",
                        79:"Cello 8'",
                    },
                    -24:{
                        67:"Praestant 32'",
                        72:"Bombarde 32'",
                    },
                    -12:{
                        68:"Principal 16'",
                        73:"Bombarde 16'",
                        74:"Fagott 16'",
                        77:"Subbass 16'",
                        78:"Violon 16'",
                    },
                    12:{
                        70:"Superoctave 4'",
                        76:"Schalmey 4'",
                    },
                    24:{
                        81:"Jubalflöte 2'",
                    }
                },
                mixtures = {
                    71:"Hintersatz 5fach",
                },
                tremulant = 86,
                couplings = {
                    82:'I an P',
                    83:'II an P',
                    84:'III an P',
                    85:'IV an P',
                }
            )
        ]
