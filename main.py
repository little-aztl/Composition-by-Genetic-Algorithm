import numpy as np
import music21 as m21

TEMPERED_SCALE_MAP = ["Rest", "F3", "F3#", "G3", "G3#", "A3", "A3#", "B3", "C4", "C4#", "D4", "D4#", "E4", "F4", "F4#", "G4", "G4#", "A4", "A4#", "B4", "C5", "C5#", "D5", "D5#", "E5", "F5", "F5#", "G5"]

PENTATONIC_SCALE_MAP = ["Rest", "G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]

MAJOR_SCALE_MAP = ["Rest", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]

MAJOR2TEMP = [0 , 3 , 5 , 7 , 8 , 10 , 12 , 13 , 15 , 17 , 19 , 20 , 22 , 24 , 25 , 27]

PENTA2TEMP = [0 , 3 , 5 , 8 , 10 , 12 , 15 , 17 , 20 , 22 , 24 , 27]

TEMP2TEMP = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27]

FITNESS_MAP = [1 , 3 , 3 , 2 , 2 , 1 , 5 , 1 , 2 , 2 , 3 , 3 , 1 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5]

CROSSOVER_POSIBILITY = 0.5
MUTATION_POSIBILITY = 0.01
POPULATION_SIZE = 10

class Pitch_Mode:
    Tempered_Scale = 0
    Pentatonic_Scale = 1
    Major_Scale = 2

SCALE_MODE = Pitch_Mode.Pentatonic_Scale
Scale_Map = []
Mode_Convert = []



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
    def Find_Prev_Note(self,cur : int):
        j = cur - 1
        while j >= 0 and self.Note_List[j].pitch == 0:
            j -= 1
        if (j >= 0):
            return j
        else:
            return None
    def Get_Not_Rest_Number(self):
        cnt = 0
        for note in self.Note_List:
            if note.pitch != 0:
                cnt += 1
        return cnt
    def Caculate(self):
        total_intervals = 0
        quality = 0
        scores = []
        effective_note_number = self.Get_Not_Rest_Number()
        if effective_note_number == 0:                                              # all breaks
            return 0,0
        if effective_note_number == 1:                                              # only one no break note
            return 1,0
        else:
            for i in range(0,self.Note_Number()):                            # start at the second note
                count_intervals = 0
                if (self.Note_List[i].pitch == 0):
                    continue
                hold = ((int)(self.Note_List[i].duration / 0.5) - 1)
                count_intervals += hold           # consider itself
                quality += hold * 1
                scores += [1 * hold]
                prev_idx = self.Find_Prev_Note(i)
                if prev_idx != None:                        # not end by j < 0
                    delta_pitch = abs(Mode_Convert[self.Note_List[i].pitch] - Mode_Convert[self.Note_List[prev_idx].pitch])
                    count_intervals += 1
                    quality += FITNESS_MAP[delta_pitch]
                    scores.append(FITNESS_MAP[delta_pitch])
                total_intervals += count_intervals
            average = quality / total_intervals
            variance = Compute_Variance(scores, average)
            return average,variance



class Piece:
    def __init__(self, ls : list[Measure] = []) -> None:
        self.fitness = 0
        self.Measure_List = ls
        self.Variance_Each_Bar = []
        self.Average_Each_Bar = []
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
    def Cross(mother, father):                         # Here may have some problems
        child = father.copy()                          # generate new piece here should be a shallow copy
        m_mother = mother.Get_Random_Measure()
        m_child = child.Get_Random_Measure()
        m_child = m_mother.copy()                      # here also should be a shallow copy
        return child
    def Calculate(self):
        variance_each_bar = []
        average_each_bar = []
        for measure_idx in range(self.Measure_Number()):
            cur_measure = self.Get_Measure(measure_idx)
            average,variance = cur_measure.Caculate()
            average_each_bar.append(average)
            variance_each_bar.append(variance)
        self.Variance_Each_Bar = variance_each_bar
        self.Average_Each_Bar = average_each_bar
    def Evaluate(self):
        self.Calculate()


def Compute_Variance(score : list[int] , average : float):
    ans = 0
    for num in score:
        ans += (num - average) * (num - average)
    ans /= len(score)
    return ans

class Population:
    def __init__(self) -> None:                        # the piece in population IS NOT the music21 format , should be converted
        self.Piece_List = []
        for i in range(POPULATION_SIZE):
            cur_seq = Produce_Initial_Duration(iteration=5)
            Produce_Initial_Pitch(cur_seq)
            self.Piece_List.append(cur_seq)


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

def Convert2Piece(score):
    p = Piece([])
    for measure in score.recurse().getElementsByClass(m21.stream.Measure):
        cur_m = Measure([])
        for note in measure.notesAndRests:
            if note.isRest:
                cur_m.Note_List.append(Note(0, note.duration.quarterLength))
            else:
                cur_m.Note_List.append(Note(Scale_Map.index(note.nameWithOctave), note.duration.quarterLength))
        p.Measure_List.append(cur_m)
    return p

if __name__ == '__main__':
    if SCALE_MODE == Pitch_Mode.Tempered_Scale:
        Scale_Map = TEMPERED_SCALE_MAP
        Mode_Convert = TEMP2TEMP
    elif SCALE_MODE == Pitch_Mode.Major_Scale:
        Scale_Map = MAJOR_SCALE_MAP
        Mode_Convert = MAJOR2TEMP
    elif SCALE_MODE == Pitch_Mode.Pentatonic_Scale:
        Scale_Map = PENTATONIC_SCALE_MAP
        Mode_Convert = PENTA2TEMP

    Init_Seq = Produce_Initial_Duration(iteration=5)
    Produce_Initial_Pitch(Init_Seq)
    Score = Convert2Music21(Init_Seq)
    Init_Seq.Evaluate()
    Score.show('midi')
