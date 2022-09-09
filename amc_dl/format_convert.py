import numpy as np
import pretty_midi as pm
import os
from argparse import ArgumentParser

EMOTION_CLASS_TO_VAL = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

################################################################################
# midi
################################################################################

def midi_to_mel_pianoroll(fn, bpm=120):
    alpha = 60 / bpm
    midi = pm.PrettyMIDI(fn)
    notes = midi.instruments[0].notes
    end_time = np.ceil(max([n.end for n in notes]) / (8 * alpha))
    pr = np.zeros((int(end_time * 32), 130))
    pr[:, -1] = 1
    for n in notes:
        s = n.start / (alpha / 4)
        e = n.end / (alpha / 4)
        p = n.pitch
        pr[int(s), int(p)] = 1
        pr[int(s) + 1: int(e) + 1, 128] = 1
        pr[int(s): int(e) + 1, -1] = 0
    pr = pr.reshape((-1, 32, 130))
    return pr


################################################################################
# melody piano-roll (T * 128)
################################################################################

def pad_mel_pianoroll(pianoroll, target_len, rest_ind=129):
    assert len(pianoroll.shape) == 2
    assert pianoroll.shape[1] == 130
    assert pianoroll.shape[0] <= target_len
    if pianoroll.shape[0] < target_len:
        pad_size = target_len - pianoroll.shape[0]
        pad_mat = np.zeros((pad_size, 130), dtype=pianoroll.dtype)
        pad_mat[:, rest_ind] = 1
        pianoroll = np.concatenate([pianoroll, pad_mat], axis=0)
    return pianoroll


def augment_mel_pianoroll(pr, shift=0):
    pitch_part = np.roll(pr[:, 0: 128], shift, axis=-1)
    control_part = pr[:, 128:]
    augmented_pr = np.concatenate([pitch_part, control_part], axis=-1)
    return augmented_pr


def mel_pianoroll_to_prmat(pr):
    steps = pr.shape[0]
    pr = pr.argmax(axis=-1)
    prmat = np.zeros((steps, 128))
    dur = 0
    for i in range(steps - 1, -1, -1):
        if pr[i] == 128:
            dur += 1
        elif pr[i] < 128:
            prmat[i, int(pr[i])] = dur + 1
            dur = 0
        else:
            dur = 0
    return prmat


def to_onehot_mel_pianoroll(x):
    pr = np.zeros((x.shape[0], 130))
    pr[np.arange(0, x.shape[0]), x.astype(int)] = 1
    return pr


def mel_pianoroll_to_notes(pr, bpm=80, begin=0., vel=100):
    prmat = mel_pianoroll_to_prmat(pr)
    notes = prmat_to_notes(prmat, bpm, begin, vel)
    return notes


################################################################################
# chord / chroma e.g., (T * 12)
################################################################################

def pad_chord_chroma(chord, target_len):
    assert len(chord.shape) == 2
    assert chord.shape[1] == 12
    assert chord.shape[0] <= target_len
    if chord.shape[0] < target_len:
        pad_size = target_len - chord.shape[0]
        pad_mat = np.zeros((pad_size, 12), dtype=chord.dtype)
        chord = np.concatenate([chord, pad_mat], axis=0)
    return chord


def augment_chord_chroma(chord, shift=0):
    augmented_chord = np.roll(chord, shift, axis=-1)
    return augmented_chord


def chord_chroma_to_notes(chroma, bpm, begin=0., velocity=80):
    alpha = 60 / bpm
    ts = [0]
    for t in range(chroma.shape[0] - 1, 0, -1):
        if (chroma[t] == chroma[t - 1]).all():
            continue
        else:
            ts.append(t)
    ts.sort()
    ets = ts[1:] + [chroma.shape[0]]
    notes = []
    for (s, e) in zip(ts, ets):
        pitches = np.where(chroma[s] != 0)[0]
        for p in pitches:
            notes.append(pm.Note(int(velocity), int(p + 48),
                                 0.25 * s * alpha + begin,
                                 0.25 * e * alpha + begin))
    return notes


################################################################################
# prmat (T * 128)  ((t, p)-position records note duration)
################################################################################

def prmat_to_notes(prmat, bpm, begin=0., vel=100):
    steps = prmat.shape[0]
    alpha = 0.25 * 60 / bpm
    notes = []
    for t in range(steps):
        for p in range(128):
            if prmat[t, p] >= 1:
                s = alpha * t + begin
                e = alpha * (t + prmat[t, p]) + begin
                notes.append(pm.Note(int(vel), int(p), s, e))
    return notes


def nmat_to_notes(nmat, bpm, begin, vel=100.):
    alpha = 0.25 * 60 / bpm
    notes = [pm.Note(int(vel), min(int(p), 127),
                     alpha * s + begin,
                     alpha * (s + d) + begin)
             for (s, p, d) in nmat[:, 0: 3]]
    return notes


def get_midi_from_notes(midi_notes):
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)

    music.instruments.append(piano)

    for note in midi_notes:
        piano.notes.append(note)

    return music


def save_as_midi_file(midi_notes, name='out'):
    music = get_midi_from_notes(midi_notes)

    music.write(name + '.mid')
    print("Saved {}.mid file".format(name))


def midi_to_nmat(fn, bpm=120):
    alpha = 60 / bpm
    midi = pm.PrettyMIDI(fn)
    notes = midi.instruments[0].notes
    length = len(notes)
    nmat = np.zeros((length, 3))
    for i, n in enumerate(notes):
        s = n.start / (alpha / 4)
        e = n.end / (alpha / 4)
        p = n.pitch
        nmat[i, 0] = s
        nmat[i, 1] = p
        nmat[i, 2] = e - s
    return nmat[np.argsort(nmat[:, 0])]


def split_to_chunks(midi_nmat, target_length=32):
    if midi_nmat.shape[0] > target_length:
        res = []
        for i in range(0, midi_nmat.shape[0], target_length):
            nmat = midi_nmat[i: i+target_length, :]
            start = np.floor(nmat[0, 0]) - 1 if np.floor(nmat[0, 0]) - 1 >= 0 else 0
            nmat = nmat - np.array([start, 0, 0])
            if nmat.shape[0] == target_length:
                res.append(nmat)
        print(np.amax(np.array(res)[:, :, 0]))
        return res


def split(midi_nmat, target_length=32):
    res = []
    if midi_nmat.shape[0] > target_length:
        nmat = np.zeros((target_length, 3))
        idx = 0
        start = 0
        added = False

        for i in range(midi_nmat.shape[0]):
            if idx == 0:
                added = False
                if i > 0:
                    start = np.floor(midi_nmat[i - 1, 0]) - 1 if np.floor(midi_nmat[i - 1, 0]) - 1 >= 0 else 0
                    nmat[0] = midi_nmat[i-1] - np.array([start, 0, 0])
                    nmat[1] = midi_nmat[i] - np.array([start, 0, 0])
                    idx += 1
                else:
                    start = np.floor(midi_nmat[i, 0]) - 1 if np.floor(midi_nmat[i, 0]) - 1 >= 0 else 0
                    nmat[0] = midi_nmat[i] - np.array([start, 0, 0])
                idx += 1
                continue
            if midi_nmat[i, 0] - start < target_length and idx < target_length:
                    nmat[idx] = midi_nmat[i] - np.array([start, 0, 0])
                    idx += 1
            else:
                if idx < target_length - 1:
                    full_nmat(nmat, idx, target_length)
                res.append(nmat)
                nmat = np.zeros((target_length, 3))
                idx = 0
                added = True

        if not added:
            if idx < target_length - 1:
                full_nmat(nmat, idx, target_length)

            res.append(nmat)
    return res


def full_nmat(nmat, idx, target_length):
    n = np.floor((target_length - 1) / idx)
    a = nmat[0:idx]
    for i in range(0, int(n)):
        ix = idx * (i + 1)
        b = np.add(a, np.array([nmat[ix - 1, 0], 0, 0]))
        b[:, 0] = [min(b[i, 0], target_length-1) for i in range(b.shape[0])]
        nmat[ix: min(ix + idx, target_length)] = b[0: min(idx, target_length - ix)]


def split_to_test_and_train(nmats, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    mask = np.random.rand(nmats.shape[0]) <= 0.9

    train_nmats = nmats[mask]
    train_length = np.array([train_nmat.shape[0] for train_nmat in train_nmats])
    test_nmats = nmats[~mask]
    test_length = np.array([test_nmat.shape[0] for test_nmat in test_nmats])
    
    np.save(os.path.join(out_path, "nmat_train.npy"), np.concatenate(train_nmats))
    np.save(os.path.join(out_path, "nmat_train_length.npy"), train_length)
    np.save(os.path.join(out_path, "nmat_val.npy"), np.concatenate(test_nmats))
    np.save(os.path.join(out_path, "nmat_val_length.npy"), test_length)


def get_nmats(midi_files: dict, path: str, target_length: int=100):
    nmats = []
    for emotion_class in EMOTION_CLASS_TO_VAL.keys():
        nmats.extend(get_nmats_for_midis(midi_files[emotion_class], emotion_class=emotion_class, path=path,
                                          target_length=target_length))
    return nmats


def get_nmats_for_midis(midi_files: list, emotion_class: str, path: str, target_length: int = 100):
    nmats = []
    for midi_file in midi_files:
        midi_nmat = midi_to_nmat(os.path.join(path, midi_file))
        for nmat in split(midi_nmat, target_length):
            nmat = np.hstack(
                (np.atleast_2d(np.full(nmat.shape[0], EMOTION_CLASS_TO_VAL[emotion_class])).T, nmat))
            nmats.append(nmat)

    return nmats


def create_emotional_data_set(path, out_path):
    midi_files = get_midi_with_emotion_classes(path)
    nmats = get_nmats(midi_files=midi_files, path=path, target_length=100)
    split_to_test_and_train(np.array(nmats), out_path)


def get_midi_with_emotion_classes(path):
    midi_files_emotion_classes = dict()
    midi_files_emotion_classes["Q1"] = [f for f in os.listdir(path) if f.endswith('.mid') and f.startswith("Q1")]
    midi_files_emotion_classes["Q2"] = [f for f in os.listdir(path) if f.endswith('.mid') and f.startswith("Q2")]
    midi_files_emotion_classes["Q3"] = [f for f in os.listdir(path) if f.endswith('.mid') and f.startswith("Q3")]
    midi_files_emotion_classes["Q4"] = [f for f in os.listdir(path) if f.endswith('.mid') and f.startswith("Q4")]

    return midi_files_emotion_classes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, help="Path to MIDI files")
    parser.add_argument("--dst_path", type=str, help="Destination path")
    args = parser.parse_args()

    create_emotional_data_set(args.src_path, args.dst_path)



