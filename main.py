import numpy as np
import music21 as m21

TEMPERED_SCALE_MAP = ["Rest", "F3", "F3#", "G3", "G3#", "A3", "A3#", "B3", "C4", "C4#", "D4", "D4#", "E4", "F4", "F4#", "G4", "G4#", "A4", "A4#", "B4", "C5", "C5#", "D5", "D5#", "E5", "F5", "F5#", "G5"]

PENTATONIC_SCALE_MAP = ["Rest", "G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]

MAJOR_SCALE_MAP = ["Rest", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]

class Pitch_Mode:
    Tempered_Scale = 0
    Pentatonic_Scale = 1
    Major_Scale = 2

SCALE_MODE = Pitch_Mode.Major_Scale
Scale_Map = []

class Note:
    def __init__(self, pitch = 0, duration = 0.5) -> None:
        self.pitch = pitch
        self.duration = duration
    def print(self):
        print(f"pitch = {Scale_Map[self.pitch]}, duration = {self.duration}")
    @staticmethod
    def Merge_Duration(note1, note2):
        return Note(note1.pitch, note1.duration + note2.duration)

def Produce_Initial_Duration(iteration = 10):
    res = []
    for i in range(32):
        res.append(Note(pitch=1))
    for step in range(iteration):
        index = np.random.randint(0, len(res) - 2)
        note1 = res[index]
        note2 = res[index + 1]
        res.pop(index + 1)
        res.pop(index)
        res.insert(index, Note.Merge_Duration(note1, note2))
    return res

def Produce_Initial_Pitch(Note_Seq):
    for note in Note_Seq:
            note.pitch = np.random.randint(len(Scale_Map))


def Note2M21(note):
    if note.pitch == 0:
        m21Rest = m21.note.Rest()
        m21Rest.duration = m21.duration.Duration(note.duration)
        return m21Rest
    m21Note = m21.note.Note(Scale_Map[note.pitch])
    m21Note.duration = m21.duration.Duration(note.duration)
    return m21Note

def Convert2Music21(Note_seq):
    score = m21.stream.Stream()
    for note in Note_seq:
        score.append(Note2M21(note))
    return score
if __name__ == '__main__':
    if SCALE_MODE == Pitch_Mode.Tempered_Scale:
        Scale_Map = TEMPERED_SCALE_MAP
    elif SCALE_MODE == Pitch_Mode.Major_Scale:
        Scale_Map = MAJOR_SCALE_MAP
    elif SCALE_MODE == Pitch_Mode.Pentatonic_Scale:
        Scale_Map = PENTATONIC_SCALE_MAP

    Init_Seq = Produce_Initial_Duration(iteration=5)
    Produce_Initial_Pitch(Init_Seq)
    Score = Convert2Music21(Init_Seq)
    Score.show()

