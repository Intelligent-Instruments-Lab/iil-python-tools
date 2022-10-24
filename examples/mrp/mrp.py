"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
TODO:
- qualities state updating
- add remaining OSC messages
"""

import mido

NOTE_ON = True
NOTE_OFF = False

class MRP(object):
    
    def __init__(self, _osc):
        self.settings = {
            'voices': {
                'max': 10, # for 10 cables
                'rule': 'oldest' # oldest, lowest, highest, quietest...
            },
            'channel': 15, # real-time midi note ch (0-indexed)
            'range': { 'start': 21, 'end': 108 } # MIDI for piano keys 0-88
        }
        self.osc = _osc
        self.pedal = {
            'damper': 0,
            'sostenuto': 0
        }
        self.program = 0 # current program (see MRP XML)
        self.notes = [] # state of each real-time midi note
        self.voices = [] # active notes indexed chronologically
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
        check if the note is off & in range
        add it as an active voice
        construct a Note On message & send over OSC
        """
        if self.note_is_off(note) == True:
            if self.note_is_in_range(note) == True:
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
                print("/mrp/midi", *m.bytes())
                # self.osc.send("/mrp/midi", *m.bytes())
                return tmp, m
            else:
                print('note_on(): note', note, 'out of range')
                return None
        else:
            print('note_on(): note', note, 'is already on')
            return None

    def note_off(self, note, velocity=0, channel=None):
        """
        check if the note is on & in range
        remove it as an active voice
        construct a Note Off message & send over OSC
        """
        if self.note_is_off(note) == False:
            if self.note_is_in_range(note) == True:
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
                print("/mrp/midi", *m.bytes())
                # self.osc.send("/mrp/midi", channel, note, velocity)
                return tmp, m
            else:
                print('note_on(): note', note, 'out of range')
                return None
        else:
            print('note_off(): note', note, 'is already off')
            return None
    
    def note_aftertouch_poly(self, note, value, channel=None):
        """
        TODO: docstring
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'polytouch',
            channel=channel,
            note=note,
            value=value
        )
        print("/mrp/midi", *m.bytes())
        # self.osc.send("/mrp/midi", *m.bytes())
    
    def note_aftertouch_channel(self, value, channel=None):
        """
        TODO: docstring
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'aftertouch',
            channel=channel,
            value=value
        )
        print("/mrp/midi")
        # self.osc.send("/mrp/midi", *m.bytes())
        
    def note_pitch_bend(self, pitch, channel=None):
        """
        TODO: docstring
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'pitchwheel',
            channel=channel,
            pitch=pitch
        )
        print("/mrp/midi", *m.bytes())
        # self.osc.send("/mrp/midi", *m.bytes())
    
    def control_change(self, controller, value, channel=None):
        """
        TODO: docstring
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'control_change',
            channel=channel,
            controller=controller,
            value=value
        )
        print("/mrp/midi", *m.bytes())
        # self.osc.send("/mrp/midi", *m.bytes())

    def program_change(self, program, channel=None):
        """
        TODO: docstring
        """
        if channel is None:
            channel = self.settings['channel']
        m = mido.Message(
            'program_change',
            channel=channel,
            program=program
        )
        print("/mrp/midi", *m.bytes())
        # self.osc.send("/mrp/midi", *m.bytes())
    
    """
    /mrp/qualities
    """
    def quality_brightness(self, note, brightness, channel=None):
        """
        brightness is an independent map to harmonic content, 
        reduced to a linear scale
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/brightness", channel, note, brightness)
        # self.osc.send("/quality/brightness", channel, note, brightness)

    def quality_intensity(self, note, intensity, channel=None):
        """
        intensity is a map to amplitude and harmonic content, 
        relative to the current intensity
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/intensity", channel, note, intensity)
        # self.osc.send("/quality/intensity", channel, note, intensity)

    def quality_pitch(self, note, pitch, channel=None):
        """
        Frequency base is relative to the fundamental frequency of the MIDI note
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/pitch", channel, note, pitch)
        # self.osc.send("/quality/pitch", channel, note, pitch)

    def quality_pitch_vibrato(self, note, pitch, channel=None):
        """
        Frequency vibrato is a periodic modulation in frequency, 
        zero-centered (+/-1 maps to range)
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/pitch/vibrato", channel, note, pitch)
        # self.osc.send("/quality/pitch/vibrato", channel=None, note, pitch)

    def quality_harmonic(self, note, harmonic, channel=None):
        """
        a single parameter that does what you hear when you shake 
        the key in the usual MRP technique (harmonic series glissando)
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/harmonic", channel, note, harmonic)
        # self.osc.send("/quality/harmonic", channel=None, note, harmonic)

    def quality_harmonics_raw(self, note, harmonics, channel=None):
        """
        a list of amplitudes for each individual harmonic, 
        which you could use to more precisely set the waveform.
        """
        if channel is None:
            channel = self.settings['channel']
        print("/quality/harmonics/raw", channel, note, harmonics)
        # self.osc.send("/quality/harmonics/raw", channel=None, note, harmonics)

    """
    /mrp/pedal
    """
    def pedal_sostenuto(self, sostenuto):
        """
        set pedal sostenuto value
        """
        self.pedal.sostenuto = sostenuto
        print("/mrp/sostenuto/damper", sostenuto)
        # self.osc.send("/mrp/pedal/sostenuto", sostenuto)

    def pedal_damper(self, damper):
        """
        set pedal damper value
        """
        self.pedal.damper = damper
        print("/mrp/pedal/damper", damper)
        # self.osc.send("/mrp/pedal/damper", damper)

    """
    /mrp/* miscellaneous
    """
    def all_notes_off(self):
        """
        turn all notes off
        TODO: reset notes and voices state
        """
        print("/mrp/allnotesoff")
        # self.osc.send("/mrp/allnotesoff")

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
        return {
            'channel': channel,
            'status': NOTE_OFF,
            'midi': {
                'number': note,
                'velocity': velocity,
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
                'harmonics_raw': 0
            }
        }

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
        if self.voices_count() < self.settings['voices']['max']
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
