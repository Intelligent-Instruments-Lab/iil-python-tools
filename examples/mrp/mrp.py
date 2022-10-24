"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
TODO:
- __init__: add settings arg
- qualities: state updating
- qualities: 'relative' arg for value updating
- qualities: update all function
- add remaining OSC messages
"""

import mido
import copy

NOTE_ON = True
NOTE_OFF = False

class MRP(object):
    
    def __init__(self, _osc):
        # settings
        self.settings = {
            'voices': {
                'max': 10, # for 10 cables
                'rule': 'oldest' # oldest, lowest, highest, quietest...
            },
            'channel': 15, # real-time midi note ch (0-indexed)
            'range': { 'start': 21, 'end': 108 } # MIDI for piano keys 0-88
        }
        # OSC reference and paths
        self.osc = _osc
        self.osc_paths = {
            'midi': '/mrp/midi',
            'qualities': {
                'brightness':    '/mrp/qualities/brightness',
                'intensity':     '/mrp/qualities/intensity',
                'pitch':         '/mrp/qualities/pitch',
                'pitch_vibrato': '/mrp/qualities/pitch/vibrato',
                'harmonic':      '/mrp/qualities/harmonic',
                'harmonics_raw': '/mrp/qualities/harmonics/raw'
            },
            'pedal': {
                'damper':    '/mrp/pedal/damper',
                'sostenuto': '/mrp/pedal/sostenuto'
            },
            'misc': {
                'allnotesoff': '/mrp/allnotesoff'
            }
        }
        # internal state
        self.notes = [] # state of each real-time midi note
        self.note = { # template note
            'channel': self.settings['channel'],
            'status': NOTE_OFF,
            'midi': {
                'number': 0,
                'velocity': 0,
                'aftertouch_poly': 0,
                'aftertouch_channel': 0,
                'pitch_bend': 0
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
        print(len(self.notes), 'notes created.')
 
    """
    /mrp/midi
    """
    def note_on(self, note, velocity, channel=None):
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
            m = mido.Message(
                'note_on',
                channel=channel,
                note=note,
                velocity=velocity
            )
            path = self.osc_paths['midi']
            print(path, 'Note On:', note, ', Velocity:', velocity)
            self.osc.send(path, *m.bytes())
            return tmp, m
        else:
            print('note_on(): invalid Note On', note)
            return None

    def note_off(self, note, velocity=0, channel=None):
        """
        check if note off is valid
        remove it as an active voice
        construct a Note Off message & send over OSC
        """
        if self.note_off_is_valid(note) == True:
            self.voices_remove(note)
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['status'] = NOTE_OFF
            tmp['channel'] = channel
            tmp['midi']['velocity'] = velocity
            m = mido.Message(
                'note_off',
                channel=channel,
                note=note,
                velocity=velocity
            )
            path = self.osc_paths['midi']
            print(path, 'Note Off:', note)
            self.osc.send(path, *m.bytes())
            return tmp, m
        else:
            print('note_off(): invalid Note Off', note)
            return None
    
    def note_aftertouch_poly(self, note, value, channel=None):
        """
        check if note message is valid
        update note aftertouch_poly state
        construct MIDI message & send over OSC
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['aftertouch_poly'] = value
            m = mido.Message(
                'polytouch',
                channel=channel,
                note=note,
                value=value
            )
            path = self.osc_paths['midi']
            print(path, 'Note Aftertouch Poly:', *m.bytes())
            self.osc.send(path, *m.bytes())
            return tmp, m
        else:
            print('note_aftertouch_poly(): invalid message')
            return None
    
    def note_aftertouch_channel(self, value, channel=None):
        """
        check if note message is valid
        update note aftertouch_channel state
        construct MIDI message & send over OSC
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['aftertouch_channel'] = value
            m = mido.Message(
                'aftertouch',
                channel=channel,
                value=value
            )
            path = self.osc_paths['midi']
            print(path, 'Note Aftertouch Channel:', *m.bytes())
            self.osc.send(path, *m.bytes())
            return tmp, m
        else:
            print('note_aftertouch_channel(): invalid message')
            return None
        
    def note_pitch_bend(self, pitch, channel=None):
        """
        check if note message is valid
        update note pitch_bend state
        construct MIDI message & send over OSC
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['pitch_bend'] = value
            m = mido.Message(
                'pitchwheel',
                channel=channel,
                pitch=pitch
            )
            path = self.osc_paths['midi']
            print(path, 'Note Pitch Bend:', *m.bytes())
            self.osc.send(path, *m.bytes())
            return tmp, m
        else:
            print('note_pitch_bend(): invalid message')
            return None
    
    def control_change(self, controller, value, channel=None):
        """
        construct MIDI CC message & send over OSC
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'control_change',
            channel=channel,
            controller=controller,
            value=value
        )
        path = self.osc_paths['midi']
        print(path, 'Control Change:', *m.bytes())
        self.osc.send(path, *m.bytes())

    def program_change(self, program, channel=None):
        """
        update program state
        construct MIDI program change message 
        & send over OSC
        """
        if channel is None:
            channel = self.settings['channel']
        self.program = program
        m = mido.Message(
            'program_change',
            channel=channel,
            program=program
        )
        path = self.osc_paths['midi']
        print(path, 'Program Change:', *m.bytes())
        self.osc.send(path, *m.bytes())
    
    """
    /mrp/qualities
    """
    def quality_brightness(self, note, brightness, channel=None):
        """
        brightness is an independent map to harmonic content, 
        reduced to a linear scale
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['brightness'] = brightness
            path = self.osc_paths['qualities']['brightness']
            print(path, channel, note, brightness)
            self.osc.send(path, channel, note, brightness)
            return tmp
        else:
            print('quality_brightness(): invalid message')
            return None

    def quality_intensity(self, note, intensity, channel=None):
        """
        intensity is a map to amplitude and harmonic content, 
        relative to the current intensity
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['intensity'] = intensity
            path = self.osc_paths['qualities']['intensity']
            print(path, channel, note, intensity)
            self.osc.send(path, channel, note, intensity)
            return tmp
        else:
            print('quality_intensity(): invalid message')
            return None

    def quality_pitch(self, note, pitch, channel=None):
        """
        Frequency base is relative to the fundamental frequency of the MIDI note
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['pitch'] = pitch
            path = self.osc_paths['qualities']['pitch']
            print(path, channel, note, pitch)
            self.osc.send(path, channel, note, pitch)
            return tmp
        else:
            print('quality_pitch(): invalid message')
            return None

    def quality_pitch_vibrato(self, note, pitch, channel=None):
        """
        Frequency vibrato is a periodic modulation in frequency, 
        zero-centered (+/-1 maps to range)
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['pitch_vibrato'] = pitch
            path = self.osc_paths['qualities']['pitch_vibrato']
            print(path, channel, note, pitch)
            self.osc.send(path, channel, note, pitch)
            return tmp
        else:
            print('quality_pitch_vibrato(): invalid message')
            return None

    def quality_harmonic(self, note, harmonic, channel=None):
        """
        a single parameter that does what you hear when you shake 
        the key in the usual MRP technique (harmonic series glissando)
        """
        if self.note_msg_is_valid(note) == True:
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['harmonic'] = harmonic
            path = self.osc_paths['qualities']['harmonic']
            print(path, channel, note, harmonic)
            self.osc.send(path, channel, note, harmonic)
            return tmp
        else:
            print('quality_harmonic(): invalid message')
            return None

    def quality_harmonics_raw(self, note, harmonics, channel=None):
        """
        a list of amplitudes for each individual harmonic, 
        which you could use to more precisely set the waveform.
        """
        if self.note_msg_is_valid(note) == True:
            if (type(harmonics) is not list):
                print('quality_harmonics_raw(): harmonics not of type List', harmonics)
                return None
            if channel is None:
                channel = self.settings['channel']
            tmp = self.notes[self.note_index(note)]
            tmp['qualities']['harmonics_raw'] = harmonics
            path = self.osc_paths['qualities']['harmonics_raw']
            print(path, channel, note, harmonics)
            self.osc.send(path, channel, note, *harmonics)
            return tmp
        else:
            print('quality_harmonics_raw(): invalid message')
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
        print(path, sostenuto)
        # self.osc.send(path, sostenuto)

    def pedal_damper(self, damper):
        """
        set pedal damper value
        """
        self.pedal.damper = damper
        path = self.osc_paths['pedal']['damper']
        print(path, damper)
        # self.osc.send(path, damper)

    """
    /mrp/* miscellaneous
    """
    def all_notes_off(self):
        """
        turn all notes off
        TODO: reset notes and voices state
        """
        print(self.osc_paths['misc']['allnotesoff'])
        # self.osc.send(self.osc_paths['misc']['allnotesoff'])

    """
    note methods
    """
    def note_create(self, note, velocity, channel=None):
        """
        TODO: match default values in mrp app
        create and return a note object
        """
        if channel is None:
            channel = self.settings['channel']
        note = copy.deepcopy(self.note)
        note['midi']['note'] = note
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
        TODO: return numbers of notes that are on
        """
        on_numbers = []
        for note in enumerate(self.notes):
            if note['status'] == NOTE_ON:
                on_numbers.append(note['midi']['number'])
        return on_numbers

    def note_on_is_valid(self, note):
        """
        check if the note is on & in range
        """
        if self.note_is_off(note) == True:
            if self.note_is_in_range(note) == True:
                return True
            else:
                print('note_on_is_valid(): note', note, 'out of range')
                return False
        else:
            print('note_on_is_valid(): note', note, 'is already on')
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
                print('note_off_is_valid(): note', note, 'out of range')
                return False
        else:
            print('note_off_is_valid(): note', note, 'is already off')
            return False

    """
    voice methods
    """
    def voices_add(self, note):
        """
        add voices up to the maximum
        then replace voices based on the rule
        """
        if note in self.voices:
            print('voices_add(): note already active')
            return self.voices
        if self.voices_count() < self.settings['voices']['max']:
            self.voices.append(note)
        else:
            rule = self.settings['voices']['rule']
            match rule:
                case 'oldest':
                    self.voices.pop(0)
                    self.voices.append(note)
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
            print('voices_note_age(): note', note, 'is off')
            return -1
