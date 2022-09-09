import torch
from curricula import Curriculum, music_curriculum
from amc_dl import format_convert
from curriculum_preset import music_corrupter_dict, default_autoenc_dict
from note_attribute_corrupter import SimpleCorrupter
from typing import Union
from note_attribute_repr import decode_atr_mat_to_emotion_nmat, NoteAttributeAutoEncoder
import numpy as np
import random
from argparse import ArgumentParser

EMOTION_LABEL_TO_NUMBER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4
}

def generate_song_notes(model, emotion: int = 1, number_of_steps: int = 1000, number_of_corrupt_steps: int = 200, target_length: int = 100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corrupter = SimpleCorrupter(**music_corrupter_dict)
    autoenc = NoteAttributeAutoEncoder(**default_autoenc_dict)

    data_in = np.zeros([1, target_length, 4])
    data_in[0, :, 0] = emotion

    for i in range(0, target_length):
        data_in[0, i, 1] = random.randint(0, 100)  # onset
        data_in[0, i, 2] = random.randint(0, 127)  # pitch
        data_in[0, i, 3] = random.randint(0, 50)  # duration

    mask = np.full((1, 100, 100), 1)
    mask = torch.from_numpy(mask.astype(np.int8)).to(device)

    corrupter.fast_mode()

    for k in range(0, number_of_steps):
        data_in, _ = autoenc.emotional_encode(data_in[0], target_length)
        if k in range(0, number_of_corrupt_steps):
            data_in, _, _, _, rel_mat = corrupter.compute_emotional_relmat_and_corrupt_atrmat_and_relmat(data_in,
                                                                                                         target_length)
        else:
            _, _, _, _, rel_mat = corrupter.compute_emotional_relmat_and_corrupt_atrmat_and_relmat(data_in,
                                                                                                   target_length)
        data_in = np.expand_dims(data_in, axis=0)
        data_in[0, :, 0] = emotion
        rel_mat = np.expand_dims(rel_mat, axis=0)
        data_in = torch.from_numpy(data_in.astype(np.int64)).to(device)
        rel_mat = torch.from_numpy(rel_mat.astype(np.int64)).to(device)
        output = model.inference(None, data_in, rel_mat, mask, None, target_length, False)
        data_in = get_result_atr_mat(output, target_length)

    result = get_result_atr_mat(output, target_length)
    nmat = decode_atr_mat_to_emotion_nmat(result[0])

    return format_convert.nmat_to_notes(nmat, bpm=120, begin=0.0)


def generate_song(curriculum: Curriculum, emotion: int = 1,
                    model_path: Union[None, str] = None, target_length: int = 100,
                    iteration_number: int = 3500, corrupt_iteration_number: int = 500,
                    out_file_name: str = "out"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    corrupter = SimpleCorrupter(**music_corrupter_dict)
    autoenc = NoteAttributeAutoEncoder(**default_autoenc_dict)

    # create data_loaders and initialize model specified by the curriculum.
    _, model = curriculum(device)

    # load a pre-trained model if necessary.
    if model_path is not None:
        model.load_model(model_path, device)

    data_in = np.zeros([1, target_length, 4])
    data_in[0, :, 0] = emotion

    for i in range(0, target_length):
        data_in[0, i, 1] = random.randint(0, 100)  # onset
        data_in[0, i, 2] = random.randint(0, 105)  # pitch
        data_in[0, i, 3] = random.randint(0, 50)  # duration

    mask = np.full((1, 100, 100), 1)
    mask = torch.from_numpy(mask.astype(np.int8)).to(device)

    corrupter.fast_mode()

    for i in range(iteration_number):
        data_in, _ = autoenc.emotional_encode(data_in[0], target_length)
        if i in range(corrupt_iteration_number):
            data_in, _, _, _, rel_mat = corrupter.compute_emotional_relmat_and_corrupt_atrmat_and_relmat(data_in,
                                                                                                         target_length)
        else:
            _, _, _, _, rel_mat = corrupter.compute_emotional_relmat_and_corrupt_atrmat_and_relmat(data_in,
                                                                                                   target_length)
        data_in = np.expand_dims(data_in, axis=0)
        data_in[0, :, 0] = emotion
        rel_mat = np.expand_dims(rel_mat, axis=0)
        data_in = torch.from_numpy(data_in.astype(np.int64)).to(device)
        rel_mat = torch.from_numpy(rel_mat.astype(np.int64)).to(device)
        output = model.inference(None, data_in, rel_mat, mask, None, target_length, False)
        data_in = get_result_atr_mat(output, target_length)

    result = get_result_atr_mat(output, target_length)
    nmat = decode_atr_mat_to_emotion_nmat(result[0])
    notes = format_convert.nmat_to_notes(nmat, bpm=120, begin=0.0)
    format_convert.save_as_midi_file(notes, out_file_name)


def get_result_atr_mat(output, target_length=100):
    result = np.zeros((1, target_length, 8), dtype=np.int64)

    for i in range(target_length):
        for j in range(0, 240, 30):
            result[0, i, int(j/30)] = np.argmax(output.cpu().detach().numpy()[0, i, j:j+30])

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model/emusebert_final.pt", help="Path to eMuseBERT model.")
    parser.add_argument("--emotion", default="Q1", type=str, choices=["Q1", "Q2", "Q3", "Q4"],
                        help="Russels 4Q model emotion label")
    parser.add_argument("--out", default="out", type=str, help="Output MIDI file name")
    parser.add_argument("--iter_num", default=3500, type=int, help="Number of eMuseBERT model iterations")
    parser.add_argument("--corrupt_iter_num", default=500, type=int, help="Number pf corruption iterations")
    args = parser.parse_args()

    generate_song(curriculum=music_curriculum, emotion=EMOTION_LABEL_TO_NUMBER[args.emotion], model_path=args.model_path,
                  target_length=100, iteration_number=args.iter_num, corrupt_iteration_number=args.corrupt_iter_num,
                  out_file_name=args.out)
