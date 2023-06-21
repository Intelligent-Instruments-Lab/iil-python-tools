"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
TODO:
- support relative updating of lists of qualities
- add qualities descriptions as comments/help
- qualities_update([note arr], qualities with arr)
- add more tests
- add timer to turn off notes after 90s
- custom max/min ranges for qualities
- harmonics_raw dict and functions
- add simulator via sc3 lib
- remove mido
- rename harmonic -> harmonic_sweep and add harmonic(note, partial, amplitude)
"""

import numpy as np
import copy

NOTE_ON = True
NOTE_OFF = False

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

class MRP(object):

    def print(self, *a, **kw):
        if self.verbose:
            print(*a, **kw)
    
    def __init__(self, _osc, settings=None, verbose=True):
        # default settings
        self.verbose = verbose
        self.settings = {
            'address': {
                'port': 7770,
                'ip': '127.0.0.1'
            },
            'voices': {
                'max': 16, # for 16 cables
                'rule': 'oldest' # oldest, lowest, highest, quietest...
            },
            'channel': 15, # real-time midi note ch (0-indexed)
            'range': { 'start': 21, 'end': 108 }, # MIDI for piano keys 0-88
            'qualities_max': 1.0,
            'qualities_min': 0.0
        }
        self.note_on_hex = 0x9F
        self.note_off_hex = 0x8F
        # custom settings
        if settings is not None:
            for k, v in settings.items():
                self.settings[k] = v
        self.print('MRP starting with settings:', self.settings)

        # OSC reference and paths
        self.osc = _osc
        self.osc_paths = {
            'midi': '/mrp/midi',
            'qualities': {
                'brightness':    '/mrp/quality/brightness',
                'intensity':     '/mrp/quality/intensity',
                'pitch':         '/mrp/quality/pitch',
                'pitch_vibrato': '/mrp/quality/pitch/vibrato',
                'harmonic':      '/mrp/quality/harmonic',
                'harmonics_raw': '/mrp/quality/harmonics/raw'
            },
            'pedal': {
                'damper':    '/mrp/pedal/damper',
                'sostenuto': '/mrp/pedal/sostenuto'
            },
            'misc': {
                'allnotesoff': '/mrp/allnotesoff'
            },
            'ui': {
                'volume':     '/ui/volume', # float vol // 0-1, >0.5 ? 4^((vol-0.5)/0.5) : 10^((vol-0.5)/0.5)
                'volume_raw': '/ui/volume/raw' # float vol // 0-1, set volume directly
            }
        }
        # internal state
        self.notes = [] # state of each real-time midi note
        self.note = { # template note
            'channel': self.settings['channel'],
            'status': NOTE_OFF,
            'midi': {
                'number': 0, # MIDI note number, not piano key number
                'velocity': 0, # not in use by MRP PLL synth
                'aftertouch_poly': 0, # not in use by MRP PLL synth
                'aftertouch_channel': 0, # not in use by MRP PLL synth
                'pitch_bend': 0 # not in use by MRP PLL synth
            },
            'qualities': {
                'brightness': 0,
                'intensity': 0,
                'pitch': 0,
                'pitch_vibrato': 0,
                'harmonic': 0,
                'harmonics_raw': []
            }
        }
        self.voices = [] # active notes indexed chronologically
        self.pedal = {
            'damper': 0,
            'sostenuto': 0
        }
        self.ui = {
            'volume': 0,
            'volume_raw': 0
        }
        self.program = 0 # current program (see MRP XML)
        # init sequence
        self.init_notes()

    def init_notes(self):
        """
        initialise an array of notes in NOTE_OFF state,
        equal in length to the number of piano keys in use
        """
        self.notes = []
        piano_keys = self.settings['range']['end'] - \
                     self.settings['range']['start'] + 1 # inclusive
        for k in range(piano_keys):
            note = self.note_create(
                self.settings['range']['start'] + k, # MIDI note numbers
                0
            )
            self.notes.append(note)
        self.print(len(self.notes), 'notes created.')
 
    """
    /mrp/midi
    """
    def note_on(self, note, velocity=1, channel=None):
        """
        check if note on is valid
        add it as an active voice
        construct a Note On message & send over OSC
        """
        if self.note_on_is_valid(note) == True:
            self.voices_add(note)
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['status'] = NOTE_ON
            tmp['channel'] = channel
            tmp['midi']['velocity'] = velocity
            path = self.osc_paths['midi']
            self.print(path, 'Note On:', note, ', Velocity:', velocity)
            self.osc.send(path, self.note_on_hex, note, velocity)
            return tmp
        else:
            self.print('note_on(): invalid Note On', note)
            return None

    def note_off(self, note, velocity=0, channel=None):
        """
        check if note off is valid
        remove it as an active voice
        construct a Note Off message & send over OSC
        """
        if self.note_off_is_valid(note) == True:
            if note in self.voices:
                self.voices_remove(note)
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['status'] = NOTE_OFF
            tmp['channel'] = channel
            tmp['midi']['velocity'] = velocity
            path = self.osc_paths['midi']
            self.print(path, 'Note Off:', note)
            self.osc.send(path, self.note_off_hex, note, velocity)
            return tmp
        else:
            self.print('note_off(): invalid Note Off', note)
            return None

    def notes_on(self, notes, velocities=None):
        vmax = self.settings['voices']['max']
        if len(notes)+1 > vmax:
            if velocities == None:
                [self.note_on(n) for n in notes]
            else:
                [self.note_on(n, velocities[i]) for i,n in enumerate(notes)]
        else:
            print('notes_on(): too many notes', notes)

    def notes_off(self, notes, channel=None):
        [self.note_off(n) for n in notes]
    
    # def control_change(self, controller, value, channel=None):
    #     """
    #     construct MIDI CC message & send over OSC
    #     """
    #     if channel is None:
    #         channel = self.settings['channel']
    #     m = mido.Message(
    #         'control_change',
    #         channel=channel,
    #         controller=controller,
    #         value=value
    #     )
    #     path = self.osc_paths['midi']
    #     self.print(path, 'Control Change:', *m.bytes())
    #     self.osc.send(path, *m.bytes())

    # def program_change(self, program, channel=None):
    #     """
    #     update program state
    #     construct MIDI program change message 
    #     & send over OSC
    #     """
    #     if channel is None:
    #         channel = self.settings['channel']
    #     self.program = program
    #     m = mido.Message(
    #         'program_change',
    #         channel=channel,
    #         program=program
    #     )
    #     path = self.osc_paths['midi']
    #     self.print(path, 'Program Change:', *m.bytes())
    #     self.osc.send(path, *m.bytes())
    
    """
    /mrp/qualities
    """
    def quality_update(self, note, quality, value, relative=False, channel=None):
        """
        Update a note's quality to a new value.

        Example
            quality_update(48, 'brightness', 0.5)

        Args
            note (int): MIDI note number
            quality (string): name of quality to update, must be same as key in osc_paths
            value (float): value of quality
            relative (bool): replace the value or add it to the current value
            channel (int): which MIDI channel to send on
        """
        if isinstance(quality, str):
            if self.note_msg_is_valid(note) == True:
                if channel is None:
                    channel = self.settings['channel']
                tmp = self.notes[self.note_index(note)]
                if isinstance(value, list) or isinstance(value, np.ndarray): # e.g. /harmonics/raw
                    if relative is True:
                        self.print('quality_update(): relative updating of lists not supported')
                        # if (len(tmp['qualities'][quality]) > 0):
                        #     for i, q in enumerate(tmp['qualities'][quality]):
                        #         tmp['qualities'][quality][i] += self.quality_clamp(value[i])
                        #         value.pop(i)
                        #     for i, v in enumerate(value):
                        #         tmp['qualities'][quality].append(value[i])
                        # else:
                        #     tmp['qualities'][quality] = [self.quality_clamp(v) for v in value]
                    else:
                        tmp['qualities'][quality] = [self.quality_clamp(v) for v in value]
                    path = self.osc_paths['qualities'][quality]
                    self.print(path, channel, note, *tmp['qualities'][quality])
                    self.osc.send(path, channel, note, *tmp['qualities'][quality])
                    return tmp
                else:
                    if relative is True:
                        tmp['qualities'][quality] = self.quality_clamp(value + tmp['qualities'][quality])
                    else:
                        tmp['qualities'][quality] = self.quality_clamp(value)
                    path = self.osc_paths['qualities'][quality]
                    self.print(path, channel, note, tmp['qualities'][quality])
                    self.osc.send(path, channel, note, tmp['qualities'][quality])
                    return tmp
            else:
                self.print('quality_update(): invalid message:', quality, note, value)
                return None
        else:
            self.print('quality_update(): "quality" is not a string:', quality)
            return None

    def qualities_update(self, note, qualities, relative=False, channel=None):
        """
        Update a note's qualities to a new set of values.

        Example
            qualities_update(48, {
                'brightness': 0.5,
                'intensity': 0.6,
                'harmonics_raw': [0.2, 0.3, 0.4]
            })
        
        Args
            note (int): MIDI note number
            qualities (dict): dict of qualities in key (string):value (float) pairs to update, 
                              must be same as key in osc_paths
            relative (bool): replace the value or add it to the current value
            channel (int): which MIDI channel to send on
        """
        if isinstance(qualities, dict):
            if self.note_msg_is_valid(note) == True:
                if channel is None:
                    channel = self.settings['channel']
                tmp = self.notes[self.note_index(note)]
                for q, v in qualities.items():
                    if isinstance(v, list) or isinstance(v, np.ndarray): # e.g. /harmonics/raw
                        if relative is True:
                            self.print('quality_update(): relative updating of lists not supported')
                        else:
                            tmp['qualities'][q] = [self.quality_clamp(i) for i in v]
                        path = self.osc_paths['qualities'][q]
                        self.print(path, channel, note, *tmp['qualities'][q])
                        self.osc.send(path, channel, note, *tmp['qualities'][q])
                    else:
                        if relative is True:
                            tmp['qualities'][q] = self.quality_clamp(v, tmp['qualities'][q])
                        else:
                            tmp['qualities'][q] = self.quality_clamp(v)
                        path = self.osc_paths['qualities'][q]
                        self.print(path, channel, note, tmp['qualities'][q])
                        self.osc.send(path, channel, note, tmp['qualities'][q])
                return tmp
            else:
                self.print('quality_update(): invalid message:', note, qualities)
                return None
        else:
            self.print('quality_update(): "qualities" is not an object:', note, qualities)
            return None

    """
    /mrp/pedal
    """
    def pedal_sostenuto(self, sostenuto):
        """
        set pedal sostenuto value
        """
        self.pedal.sostenuto = sostenuto
        path = self.osc_paths['pedal']['sostenuto']
        self.print(path, sostenuto)
        self.osc.send(path, sostenuto)

    def pedal_damper(self, damper):
        """
        set pedal damper value
        """
        self.pedal.damper = damper
        path = self.osc_paths['pedal']['damper']
        self.print(path, damper)
        self.osc.send(path, damper)

    """
    /mrp/* miscellaneous
    """
    def all_notes_off(self):
        """
        turn all notes off
        """
        path = self.osc_paths['misc']['allnotesoff']
        self.print(path)
        self.osc.send(path)
        self.init_notes()
        self.voices_reset()

    """
    /mrp/ui
    """
    def ui_volume(self, value):
        """
        float vol // 0-1, >0.5 ? 4^((vol-0.5)/0.5) : 10^((vol-0.5)/0.5)
        """
        self.ui.volume = value
        path = self.osc_paths['ui']['volume']
        self.print(path, value)
        self.osc.send(path, value)

    def ui_volume_raw(self, value):
        """
        float vol // 0-1, set volume directly
        """
        self.ui.volume_raw = value
        path = self.osc_paths['ui']['volume_raw']
        self.print(path, value)
        self.osc.send(path, value)

    """
    note methods
    """
    def note_create(self, n, velocity, channel=None):
        """
        create and return a note object
        """
        if channel is None:
            channel = self.settings['channel']
        note = copy.deepcopy(self.note)
        note['midi']['number'] = n
        note['midi']['velocity'] = velocity
        return note

    def note_is_in_range(self, note):
        """
        check if a note is in valid range
        """
        start = self.settings['range']['start']
        end = self.settings['range']['end']
        if start > note or note > end:
            return False
        return True

    def note_is_off(self, note):
        """
        check if a note is off
        """
        index = note - self.settings['range']['start']
        if self.notes[index]['status'] == NOTE_ON:
            return False
        return True

    def note_index(self, note):
        return note - self.settings['range']['start']

    def note_on_numbers(self):
        """
        return numbers of notes that are on
        """
        return [
            note['midi']['number'] 
            for note in self.notes 
            if note['status']==NOTE_ON]
# 
    def note_on_is_valid(self, note):
        """
        check if the note is on & in range
        """
        if self.note_is_in_range(note) == True:
            if self.note_is_off(note) == True:
                return True
            else:
                self.print('note_on_is_valid(): note', note, 'is already on')
                return False
        else:
            self.print('note_on_is_valid(): note', note, 'out of range')
            return False
            

    def note_msg_is_valid(self, note):
        return self.note_off_is_valid(note)

    def note_off_is_valid(self, note):
        """
        check if the note is off & in range
        """
        if self.note_is_off(note) == False:
            if self.note_is_in_range(note) == True:
                return True
            else:
                self.print('note_off_is_valid(): note', note, 'out of range')
                return False
        else:
            self.print('note_off_is_valid(): note', note, 'is already off')
            return False

    """
    qualities methods
    """
    def quality_clamp(self, value):
        ### NOTE pitch, at least, can be negative or > 1
        return float(value)
        # return float(clamp(value, self.settings['qualities_min'], self.settings['qualities_max']))

    """
    voice methods
    """
    def voices_add(self, note):
        """
        add voices up to the maximum
        then replace voices based on the rule
        """
        if note in self.voices:
            self.print('voices_add(): note already active')
            return self.voices
        if self.voices_count() < self.settings['voices']['max']:
            self.voices.append(note)
        else:
            rule = self.settings['voices']['rule']
            match rule:
                case 'oldest':
                    oldest = self.voices[0]
                    self.print('voices_add(): removing oldest', oldest)
                    self.voices.pop(0)
                    self.voices.append(note)
                    self.note_off(oldest)
                    return self.voices
                case _: # lowest, highest, quietest, ...
                    return self.voices
        return self.voices

    def voices_remove(self, note):
        self.voices.remove(note)
        return self.voices

    def voices_update(self):
        """
        reconstruct active voices list based on self.notes
        """
        self.voices = self.note_on_numbers()
        return self.voices

    def voices_compare(self):
        """
        check if voices and notes match
        """
        note_on_numbers = self.note_on_numbers()
        return note_on_numbers == self.voices, {'notes': note_on_numbers}, {'voices': self.voices}

    def voices_reset(self):
        self.voices = []

    def voices_count(self):
        return len(self.voices)

    def voices_position(self, note):
        """
        return position of a note in voice queue
        """
        if note in self.voices:
            return self.voices.index(note)
        else:
            self.print('voices_note_age(): note', note, 'is off')
            return -1

    """
    misc methods
    """
    def cleanup(self):
        print('MRP exiting...')
        self.all_notes_off()
