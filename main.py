import numpy as np
import music21 as m21

TEMPERED_SCALE_MAP = ["Rest", "F3", "F3#", "G3", "G3#", "A3", "A3#", "B3", "C4", "C4#", "D4", "D4#", "E4", "F4", "F4#", "G4", "G4#", "A4", "A4#", "B4", "C5", "C5#", "D5", "D5#", "E5", "F5", "F5#", "G5"]

PENTATONIC_SCALE_MAP = ["Rest", "G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]

MAJOR_SCALE_MAP = ["Rest", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]

class Pitch_Mode:
    Tempered_Scale = 0
    Pentatonic_Scale = 1
    Major_Scale = 2

SCALE_MODE = Pitch_Mode.Tempered_Scale
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
    def Mutate(self):
        self.pitch = np.random.randint(len(Scale_Map))

class Measure:
    def __init__(self, ls : list[Note] = []) -> None:
        self.Note_List = ls
    def Note_Number(self) -> int:
        return len(self.Note_List)
    def At(self, idx : int) -> Note:
        return self.Note_List[idx]
    def Transposition(self, delta = None):
        if not delta:
            delta = np.random.randint()

class Piece:
    def __init__(self, ls : list[Measure] = []) -> None:
        self.Measure_List = ls
    def Measure_Number(self) -> int:
        return len(self.Measure_List)
    def Get_Measure(self, idx : int) -> Measure:
        return self.Measure_List[idx]
    def Get_Note(self, idx : int) -> Note:
        cnt = 0
        for measure in self.Measure_List:
            if cnt + measure.Note_Number() > idx:
                return measure.At(idx - cnt)
            cnt += measure.Note_Number()
        return None
    def Flatten(self) -> list[Note]:
        ret = []
        for measure in self.Measure_List:
            ret += measure.Note_List
        return ret
    def Get_Random_Measure(self):
        measure_idx = np.random.randint(0, self.Measure_Number() - 1)
        current_measure = self.Get_Measure(measure_idx)
        return current_measure
    def Get_Random_Note(self):
        current_measure = self.Get_Random_Measure()
        while current_measure.Note_Number == 1:
            current_measure = self.Get_Random_Measure()
        note_idx = np.random.randint(0, current_measure.Note_Number() - 2)
        return current_measure.At(note_idx), current_measure, note_idx
    def Mutate(self):
        n,_,_ = self.Get_Random_Note()
        n.Mutate()
    def Cross(p1, p2):
        m1 = p1.Get_Random_Measure()
        m2 = p2.Get_Random_Measure()
        m1, m2 = m2, m1


def Produce_Initial_Duration(measure_number = 4, iteration = 10) -> Piece:
    ret = Piece()
    for i in range(measure_number):
        current_measure = []
        for j in range(8):
            current_measure.append(Note(pitch=1))
        ret.Measure_List.append(Measure(current_measure))
    for step in range(iteration):
        _, current_measure, note_idx = ret.Get_Random_Note()
        n1 = current_measure.At(note_idx)
        n2 = current_measure.At(note_idx + 1)
        current_measure.Note_List.pop(note_idx + 1)
        current_measure.Note_List.pop(note_idx)
        current_measure.Note_List.insert(note_idx, Note.Merge_Duration(n1, n2))

    return ret

def Produce_Initial_Pitch(piece : Piece):
    for note in piece.Flatten():
            note.Mutate()


def Note2M21(note : Note):
    if note.pitch == 0:
        m21Rest = m21.note.Rest()
        m21Rest.duration = m21.duration.Duration(note.duration)
        return m21Rest
    m21Note = m21.note.Note(Scale_Map[note.pitch])
    m21Note.duration = m21.duration.Duration(note.duration)
    return m21Note

def Convert2Music21(piece : Piece):
    score = m21.stream.Stream()
    for note in piece.Flatten():
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

