import numpy as np
import music21 as m21
import math
import copy

TEMPERED_SCALE_MAP = ["Rest", "F3", "F3#", "G3", "G3#", "A3", "A3#", "B3", "C4", "C4#", "D4", "D4#", "E4", "F4", "F4#", "G4", "G4#", "A4", "A4#", "B4", "C5", "C5#", "D5", "D5#", "E5", "F5", "F5#", "G5"]
PENTATONIC_SCALE_MAP = ["Rest", "G3", "A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]
MAJOR_SCALE_MAP = ["Rest", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]

class Converter:
    MAJOR2TEMP = [0 , 3 , 5 , 7 , 8 , 10 , 12 , 13 , 15 , 17 , 19 , 20 , 22 , 24 , 25 , 27]
    PENTA2TEMP = [0 , 3 , 5 , 8 , 10 , 12 , 15 , 17 , 20 , 22 , 24 , 27]
    @staticmethod
    def Convert2Tempered(pitch : int) -> int:
        if SCALE_MODE == Pitch_Mode.Pentatonic_Scale:
            return Converter.PENTA2TEMP[pitch]
        elif SCALE_MODE == Pitch_Mode.Major_Scale:
            return Converter.MAJOR2TEMP[pitch]
        else:
            return pitch
    @staticmethod
    def Note2M21(note):
        if note.pitch == 0:
            m21Rest = m21.note.Rest()
            m21Rest.duration = m21.duration.Duration(note.duration)
            return m21Rest
        m21Note = m21.note.Note(Scale_Map[note.pitch])
        m21Note.duration = m21.duration.Duration(note.duration)
        return m21Note
    @staticmethod
    def Convert2Music21(piece):
        score = m21.stream.Stream()
        score.append(m21.instrument.Harp())
        for note in piece.Flatten():
            score.append(Converter.Note2M21(note))
        return score
    @staticmethod
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
    @property
    def Standard_Pitch(self):
        return Converter.Convert2Tempered(self.pitch)
    def print(self):
        print(f"pitch = {Scale_Map[self.pitch]}, duration = {self.duration}")
    @staticmethod
    def Merge_Duration(note1, note2):
        return Note(note1.pitch, note1.duration + note2.duration)
    def Randomly_Change_Pitch(self):
        self.pitch = np.random.randint(0, len(Scale_Map))
    def Split(self):
        if self.duration <= 0.5:
            return None, None
        tmp = int(self.duration * 4)
        tmp //= 2
        tmp /= 4
        return Note(self.pitch, tmp), Note(self.pitch, self.duration - tmp)

class Measure:
    def __init__(self, ls : list[Note] = []) -> None:
        self.Note_List = ls
    @property
    def Note_Number(self) -> int:
        return len(self.Note_List)
    def At(self, idx : int) -> Note:
        return self.Note_List[idx]
    def Get_Random_Note_Index(self):
        note_idx = np.random.randint(0, self.Note_Number)
        return note_idx
    def Mutate(self, mutate_type = None):
        if not mutate_type:
            mutate_type = np.random.randint(1, 4)
        if mutate_type == 1: # Randomly choose a note and randomly change its pitch
            note_idx = self.Get_Random_Note_Index()
            self.At(note_idx).Randomly_Change_Pitch()
        elif mutate_type == 2: # Randomly choose a note and split it
            note_idx = self.Get_Random_Note_Index()
            note1, note2 = self.At(note_idx).Split()
            if not note1:
                return False
            self.Note_List.pop(note_idx)
            self.Note_List.insert(note_idx, note2)
            self.Note_List.insert(note_idx, note1)
        elif mutate_type == 3: # Randomly choose two consecutive notes and merge them
            if self.Note_Number == 1:
                return False
            note_idx = 0
            while True:
                note_idx = self.Get_Random_Note_Index()
                if note_idx < self.Note_Number - 1:
                    break
            p1 = self.At(note_idx).pitch
            d1 = self.At(note_idx).duration
            d2 = self.At(note_idx + 1).duration
            self.Note_List.pop(note_idx)
            self.Note_List.pop(note_idx)
            self.Note_List.insert(note_idx, Note(p1, d1 + d2))
        return True
    def Transform(self):
        mutate_type = np.random.randint(1, 4)
        if mutate_type == 1: # Reverse the whole measure
            self.Note_List.reverse()
        elif mutate_type == 2: # Reflection
            pitches = [note.pitch for note in self.Note_List if note.pitch > 0]
            if len(pitches) == 0:
                return False
            lower_bound = min(pitches)
            upper_bound = max(pitches)
            lower_bound2 = (int)(math.ceil((1 + upper_bound) / 2))
            upper_bound2 = (lower_bound + len(Scale_Map) - 1) // 2
            lower_bound = max(lower_bound, lower_bound2)
            upper_bound = min(upper_bound, upper_bound2)
            if lower_bound > upper_bound:
                return False
            center = np.random.randint(lower_bound, upper_bound + 1)
            for n in self.Note_List:
                if n.pitch > 0:
                    n.pitch = center * 2 - n.pitch
        elif mutate_type == 3: # Transposition
            pitches = [note.pitch for note in self.Note_List if note.pitch > 0]
            if (len(pitches) == 0):
                return False
            lower_bound = 1 - min(pitches)
            upper_bound = len(Scale_Map) - 1 - max(pitches)
            lower_bound = max(lower_bound, -4) # consecutive assumption
            upper_bound = min(upper_bound, 4)
            t = np.random.randint(lower_bound, upper_bound + 1)
            for n in self.Note_List:
                if n.pitch > 0:
                    n.pitch += t
        return True

class Gaussian:
    def normalized_gauss(miu, sigma, x):
        return np.exp(-(x - miu) ** 2 / 2 / sigma ** 2)

class Piece:
    def __init__(self) -> None:
        self.fitness = 0
        self.Measure_List = []
    def Measure_Number(self) -> int:
        return len(self.Measure_List)
    def Get_Measure(self, idx : int) -> Measure:
        return self.Measure_List[idx]
    def Get_Note(self, idx : int) -> Note:
        cnt = 0
        for measure in self.Measure_List:
            if cnt + measure.Note_Number > idx:
                return measure.At(idx - cnt)
            cnt += measure.Note_Number
        return None
    def Flatten(self) -> list[Note]:
        ret = []
        for measure in self.Measure_List:
            ret += measure.Note_List
        return ret
    def Get_Random_Measure(self):
        measure_idx = np.random.randint(0, self.Measure_Number())
        current_measure = self.Get_Measure(measure_idx)
        return current_measure

    def Mutate(self):
        return self.Get_Random_Measure().Mutate()
    def Transform(self):
        return self.Get_Random_Measure().Transform()

    def Compute_Features(self):
        pitches = [note.Standard_Pitch for note in self.Flatten() if note.Standard_Pitch > 0]
        durations = [note.duration for note in self.Flatten()]
        intervals = [pitch_pair[1] - pitch_pair[0] for pitch_pair in zip(pitches[:-1], pitches[1:])]
        intervals_count = len(intervals)
        rest_length = sum([note.duration for note in self.Flatten() if note.pitch == 0])
        if intervals_count <= 15 or rest_length >= 2:
            self.pitch_range = -100
            self.dissonant_intervals = -100
            self.contour_direction = -100
            self.contour_stability = -100
            self.rhythmic_variery = -100
            self.rhythmic_range = -100
            return
        def Compute_Pitch_Range():
            max_pitch = max(pitches)
            min_pitch = min(pitches)
            return min(1, (max_pitch - min_pitch) / 24)
        def Compute_Dissonant_Intervals():
            second_dissonant_intervals_count = len([1 for interval in intervals if abs(interval) in [6, 11] or abs(interval) >= 12])
            first_dissonant_intervals_count = len([1 for interval in intervals if abs(interval) >= 8 and abs(interval) <= 10])
            return (first_dissonant_intervals_count * 0.5 + second_dissonant_intervals_count) / intervals_count
        def Compute_Contour_Direction():
            rising_intervals_count = len([1 for inte in intervals if inte > 0])
            return rising_intervals_count / intervals_count
        def Compute_Coutour_Stability():
            current_moving_direction = 0
            if intervals[0] > 0:
                current_moving_direction = 1
            elif intervals == 0:
                current_moving_direction = 0
            else:
                current_moving_direction = -1
            consecutive_intervals_count = 0
            for i in range(1, intervals_count):
                if intervals[i] > 0:
                    if current_moving_direction == 1:
                        consecutive_intervals_count += 1
                    current_moving_direction = 1
                elif intervals[i] == 0:
                    if current_moving_direction == 0:
                        consecutive_intervals_count += 1
                    current_moving_direction = 0
                else:
                    if current_moving_direction == -1:
                        consecutive_intervals_count += 1
                    current_moving_direction = -1
            return consecutive_intervals_count / (intervals_count - 1)
        def Compute_Rhythmic_Variety():
            return np.unique(durations).shape[0] / 16
        def Compute_Rhythmic_Range():
            return max(durations) / min(durations) / 16
        self.pitch_range = Compute_Pitch_Range()
        self.dissonant_intervals = Compute_Dissonant_Intervals()
        self.contour_direction = Compute_Contour_Direction()
        self.contour_stability = Compute_Coutour_Stability()
        self.rhythmic_variery = Compute_Rhythmic_Variety()
        self.rhythmic_range = Compute_Rhythmic_Range()

    def Evalute(self):
        self.score = 0
        self.score += Gaussian.normalized_gauss(0.30, 0.11, self.pitch_range)
        self.score += Gaussian.normalized_gauss(0.01, 0.02, self.dissonant_intervals)
        self.score += Gaussian.normalized_gauss(0.49, 0.06, self.contour_direction)
        self.score += Gaussian.normalized_gauss(0.40, 0.11, self.contour_stability)
        self.score += Gaussian.normalized_gauss(0.24, 0.07, self.rhythmic_variery)
        self.score += Gaussian.normalized_gauss(0.32, 0.11, self.rhythmic_range)
        self.score += self.Flatten()[-1].duration / 4
        if SCALE_MODE != Pitch_Mode.Tempered_Scale:
            self.score += (int)(Converter.Convert2Tempered(self.Flatten()[-1].pitch) in [20]) / 2


def Produce_Initial_Duration(measure_number = 4, iteration = 10) -> Piece:
    ret = Piece()
    for i in range(measure_number):
        current_measure = []
        for j in range(8):
            current_measure.append(Note(pitch=1))
        ret.Measure_List.append(Measure(current_measure))
    for step in range(iteration):
        ret.Get_Random_Measure().Mutate(mutate_type=3)
    return ret

def Produce_Initial_Pitch(piece : Piece):
    for note in piece.Flatten():
            note.Randomly_Change_Pitch()


def Selection(pieces : list[Piece], threshold = 20) -> list[Piece]:
    for p in pieces:
        p.Compute_Features()
        p.Evalute()
    pieces.sort(key=lambda p : p.score, reverse=True)
    return pieces[:threshold]

def ChildBearing(pieces : list[Piece], mutation_volume = 10, transformation_volume = 3, crossover_volume = 1) -> list[Piece]:
    pieces_count = len(pieces)
    children = []
    for step in range(mutation_volume):
        piece_idx = np.random.randint(0, pieces_count)
        new_piece = copy.deepcopy(pieces[piece_idx])
        new_piece.Mutate()
        children.append(new_piece)
    for step in range(transformation_volume):
        piece_idx = np.random.randint(0, pieces_count)
        new_piece = copy.deepcopy(pieces[piece_idx])
        new_piece.Transform()
        children.append(new_piece)
    for step in range(crossover_volume):
        piece_idx1 = np.random.randint(0, pieces_count)
        piece_idx2 = 0
        while True:
            piece_idx2 = np.random.randint(0, pieces_count)
            if piece_idx1 != piece_idx2:
                break
        new_piece1 = copy.deepcopy(pieces[piece_idx1])
        new_piece2 = copy.deepcopy(pieces[piece_idx2])
        m1 = new_piece1.Get_Random_Measure()
        m2 = new_piece2.Get_Random_Measure()
        m1, m2 = m2, m1
        children.append(new_piece1)
        children.append(new_piece2)
    return children
    # return pieces

def Initialize_Population(volume = 10) -> list[Piece]:
    pieces = [Produce_Initial_Duration(iteration=0) for step in range(volume)]
    for p in pieces:
        Produce_Initial_Pitch(p)
    return pieces

def Evolution(iteration = 500):
    population = Initialize_Population()
    for step in range(iteration):
        print(f"step = {step}", end = " ")
        current_population = population
        for j in range(100):
            current_population = ChildBearing(current_population)
            population += current_population
        population = Selection(population)
        print("score = %.2lf" % population[0].score)
    yield population[0]


if __name__ == '__main__':
    if SCALE_MODE == Pitch_Mode.Tempered_Scale:
        Scale_Map = TEMPERED_SCALE_MAP
    elif SCALE_MODE == Pitch_Mode.Major_Scale:
        Scale_Map = MAJOR_SCALE_MAP
    elif SCALE_MODE == Pitch_Mode.Pentatonic_Scale:
        Scale_Map = PENTATONIC_SCALE_MAP

    for piece in Evolution():
        s = Converter.Convert2Music21(piece)
        s.show('midi')
        s.write('midi', 'result.mid')
