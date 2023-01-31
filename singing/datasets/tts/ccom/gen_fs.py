import os

os.environ["OMP_NUM_THREADS"] = "1"

from datasets.tts.utils import build_phone_encoder
from utils.indexed_datasets import IndexedDatasetBuilder
import glob
import json
import logging
import sys
import traceback
from multiprocessing.pool import Pool
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.hparams import hparams, set_hparams
from utils.preprocessor import process_utterance, get_pitch, get_mel2ph
import pdb


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def read_lab_file(lab_fn, note2midi, hparams):
    """
    Args:
        lab_fn (str):
        note2midi (dict):
    Return:
        str:
        str:
        str:
    """
    def quantify(x):
        x = int(float(x) * 1e-7 * hparams['audio_sample_rate'] / hparams['hop_size'])
        return x
    lab = open(lab_fn, 'r').readlines()
    phs, durs, pitchs = list(), list(), list()
    for item in lab:
        phone, start, end, note = item.strip().split()
        dur = str(quantify(end) - quantify(start))
        midi = note2midi[note]
        
        phs += [phone]
        durs += [dur]
        pitchs += [midi]

    return ' '.join(phs), ' '.join(durs), ' '.join(pitchs)

def dur_to_mel2ph(dur, mel, hparams):
    """
    Args:
        dur (str): 
        mel (ndarray):
    """
    dur_list = dur.strip().split()
    dur_list[-1] = len(mel)
    split = [int(item) for item in dur_list]
    split = [0] + split
    split = np.cumsum(split)
    mel2ph = np.zeros([mel.shape[0]], np.int)
    for ph_idx in range(len(dur_list)):
        mel2ph[split[ph_idx]:split[ph_idx + 1]] = ph_idx + 1
    mel2ph_torch = torch.from_numpy(mel2ph)
    T_t = len(dur_list)
    dur_ = mel2ph_torch.new_zeros([T_t + 1]).scatter_add(0, mel2ph_torch, torch.ones_like(mel2ph_torch))
    dur_ = dur_[1:].numpy()
    return mel2ph, dur_

def load_note2midi_dict(note2midi_dict_fn):
    """
    Args:
       note2midi_dict_fn (str):
    Retuen:
        dict:  
    """
    note2midi = json.load(open(note2midi_dict_fn, 'r'))
    return note2midi

def quantify_dur(note_durs, note_phones, hparams):
    """
    Args:
        note_durs (list): 
        note_phones (list):
    Returns:
        (list): quantifed_note_durs
    """
    def quantify(x, bpm=124):
        frames16 = (60 / bpm / 4) * hparams['audio_sample_rate'] / hparams['hop_size']
        x = round(x / frames16) * frames16
        return int(x)
    quantifed_note_durs = list()
    cnt = 0
    cumsum_dur = 0
    for note_phone, note_dur in zip(note_phones, note_durs):
        if note_phone in ['sil', 'sp', '<UNK>', '<EOS>']:
            quantifed_note_durs += [quantify(int(note_dur))]
        else:
            cnt += 1
            if cnt == 2:
                cumsum_dur += int(note_dur)
                quantifed_note_durs += [quantify(cumsum_dur), quantify(cumsum_dur)]
                cumsum_dur = 0
                cnt = 0
            else:
                cumsum_dur += int(note_dur)

    return quantifed_note_durs

def process_item(stats_data_dir, encoder, note2midi, wav_fn, lab_fn):
    item_name = os.path.splitext(os.path.basename(wav_fn))[0]   # eg: item_name='00001'
    item_tag = os.path.dirname(wav_fn).split('/')[-1][0]
    item_name = f"{item_tag}_{item_name}"
    spk_id = 0
    ph, dur, midi = read_lab_file(lab_fn, note2midi, hparams)
    ph = "<UNK> " + ph + " <EOS>"
    dur = "0 " + dur + " 0"
    midi = "0 " + midi + " 0"
    try:
        phone_encoded = encoder.encode(ph)
        wav_data, mel = process_utterance(
            wav_fn, fft_size=hparams['n_fft'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=False, vocoder=hparams['vocoder'])
        mel = mel.T  # [T, 80]
    except:
        traceback.print_exc()
        print("| invalid data", item_name)
        return None
    mel2ph, note_dur = dur_to_mel2ph(dur, mel, hparams)
    f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
    note_pitch = np.array(midi.split(), dtype=np.int)
    note_dur, note_pitch = list(note_dur), list(note_pitch)
    quantifed_note_dur = quantify_dur(note_dur, ph.strip().split(), hparams)
    # if max(quantifed_note_dur) > 200:
    #    print(item_name, max(quantifed_note_dur))
    #    pdb.set_trace()
    assert len(phone_encoded) == len(note_pitch) == len(quantifed_note_dur), \
            f"len(phone_encoded): {len(phone_encoded)}, len(note_pitch): {len(note_pitch)}, len(quantifed_note_dur): {len(quantifed_note_dur)}"

    return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, quantifed_note_dur, note_pitch


def process_data(stats_data_dir, encoder, wav_fns, lab_fns, data_dir, prefix):
    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = list()

    note2midi = load_note2midi_dict(f"{stats_data_dir}/note_midi.json")
    for wav_fn, lab_fn in zip(wav_fns, lab_fns):  # eg: wav_fn='1-48khz/00001.wav'; lab_fn='1-lab/00001.lab'
        wav_item_name = os.path.splitext(os.path.basename(wav_fn))[0]   # eg: item_name='00001'
        lab_item_name = os.path.splitext(os.path.basename(lab_fn))[0]   # eg: item_name='00001'
        assert wav_item_name == lab_item_name, f"wav_fn: {wav_fn}, lab_fn: {lab_fn}"
        futures.append(p.apply_async(process_item, args=(stats_data_dir, encoder, note2midi, wav_fn, lab_fn)))
        # futures.append(process_item(stats_data_dir, encoder, note2midi, wav_fn, lab_fn))
    p.close()

    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    all_keys = []
    lengths = []
    f0s = []
    durs = []
    for future in tqdm(futures):
        res = future.get()
        if res is None:
            continue
        item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur, note_dur, note_pitch = res
        # skip outliers
        if max(note_dur) > 500-2:
            print(f"filter {item_name}")
            continue
        item_name = f'ccom_{item_name}'
        builder.add_item({
            'item_name': item_name,
            'txt': item_name,
            'phone': phone_encoded,
            'mel': mel,
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            'pitch': pitch,
            'f0': f0,
            'note_dur': note_dur,
            'note_pitch': note_pitch,
        })
        lengths.append(mel.shape[0])
        all_keys.append(item_name)
        f0s.append(f0)
        durs.append(dur)
    p.join()
    builder.finalize()
    np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
    np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
    np.save(f'{data_dir}/{prefix}_f0s.npy', f0s)
    np.save(f'{data_dir}/{prefix}_durs.npy', durs)

if __name__ == "__main__":
    set_hparams()
    raw_data_dir = hparams['raw_data_dir']      # CCOM_22k
    stats_data_dir = hparams['stats_data_dir']
    all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/*/*.wav'))
    all_lab_fns = sorted(glob.glob(f'{raw_data_dir}/*/*.lab'))
    assert len(all_wav_fns) == len(all_lab_fns), f"num wav: {len(all_wav_fns)}, num lab: {len(all_lab_fns)}"
    logging.info("train {}".format(len(all_wav_fns)))

    ph_set = [x.strip() for x in open(f'{stats_data_dir}/phone_set.txt').readlines()]
    print(ph_set)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
    encoder = build_phone_encoder(hparams['data_dir'])

    # encoder = build_phone_encoder(raw_data_dir)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    test_num = 10
    process_data(stats_data_dir, encoder, all_wav_fns[:test_num], all_lab_fns[:test_num], hparams['data_dir'], 'valid')
    process_data(stats_data_dir, encoder, all_wav_fns[:test_num], all_lab_fns[:test_num], hparams['data_dir'], 'test')
    process_data(stats_data_dir, encoder, all_wav_fns[test_num:], all_lab_fns[test_num:], hparams['data_dir'], 'train')

