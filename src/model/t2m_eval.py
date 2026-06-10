"""
T2M evaluator models — architecture and text preprocessing from Guo et al. (2022)
"Generating Diverse and Natural 3D Human Motions from Text."

The POS_enumerator, VIP word lists, and WordVectorizer below are copied verbatim
from the original repo so that text embeddings match what the evaluator was
trained on. Text is NOT re-tokenised here — we use the pre-tagged `word/POS`
tokens already stored in the HumanML3D text files.

Motions passed to encode_motion() must be normalised with the T2M evaluator's
own Mean/Std (Comp_v6_KLD01/meta), not the HumanML3D training Mean/Std.

Required files (see evaluate.py --help):
  <evaluator_dir>/checkpoint/finest.tar
  <evaluator_dir>/glove/our_vab_data.npy
  <evaluator_dir>/glove/our_vab_words.pkl
  <evaluator_dir>/glove/our_vab_idx.pkl
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ── Text preprocessing tables (verbatim from EricGuo5513/text-to-motion) ────────

POS_enumerator = {
    'VERB': 0, 'NOUN': 1, 'DET': 2, 'ADP': 3, 'NUM': 4, 'AUX': 5,
    'PRON': 6, 'ADJ': 7, 'ADV': 8, 'Loc_VIP': 9, 'Body_VIP': 10,
    'Obj_VIP': 11, 'Act_VIP': 12, 'Desc_VIP': 13, 'OTHER': 14,
}

Loc_list  = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise',
             'forward', 'back', 'backward', 'up', 'down', 'straight', 'curve')
Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg',
             'waist', 'eye', 'knee', 'shoulder', 'thigh')
Obj_List  = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball',
             'handrail', 'baseball', 'basketball')
Act_list  = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat',
             'throw', 'hop', 'dance', 'jump', 'turn', 'stumble', 'dance', 'stop',
             'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
             'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate',
             'spin', 'spread', 'climb')
Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy',
             'angry', 'sad', 'happily', 'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP':  Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP':  Obj_List,
    'Act_VIP':  Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer:
    def __init__(self, glove_dir, prefix='our_vab'):
        vectors  = np.load(os.path.join(glove_dir, f'{prefix}_data.npy'))
        words    = pickle.load(open(os.path.join(glove_dir, f'{prefix}_words.pkl'), 'rb'))
        word2idx = pickle.load(open(os.path.join(glove_dir, f'{prefix}_idx.pkl'), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        vec[POS_enumerator.get(pos, POS_enumerator['OTHER'])] = 1.0
        return vec

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word].astype(np.float32)
            vip_pos = next((k for k, vals in VIP_dict.items() if word in vals), None)
            pos_vec = self._get_pos_ohot(vip_pos if vip_pos is not None else pos)
        else:
            word_vec = self.word2vec['unk'].astype(np.float32)
            pos_vec  = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec


# ── Evaluator networks ──────────────────────────────────────────────────────────

class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(_init_weight)
        self.out_net.apply(_init_weight)

    def forward(self, x):                                       # (B, T, D)
        out = self.main(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T//4, D')
        return self.out_net(out)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.input_emb   = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net  = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden = nn.Parameter(torch.randn(2, 1, hidden_size))

    def forward(self, inputs, m_lens):
        B = inputs.shape[0]
        emb = pack_padded_sequence(self.input_emb(inputs), m_lens.cpu(),
                                   batch_first=True, enforce_sorted=False)
        _, last = self.gru(emb, self.hidden.repeat(1, B, 1))
        return self.output_net(torch.cat([last[0], last[1]], dim=-1))


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super().__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.pos_emb     = nn.Linear(pos_size, word_size)
        self.input_emb   = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net  = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.pos_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden = nn.Parameter(torch.randn(2, 1, hidden_size))

    def forward(self, word_embs, pos_onehot, cap_lens):
        B = word_embs.shape[0]
        inputs = self.input_emb(word_embs + self.pos_emb(pos_onehot))
        emb = pack_padded_sequence(inputs, cap_lens.cpu(),
                                   batch_first=True, enforce_sorted=False)
        _, last = self.gru(emb, self.hidden.repeat(1, B, 1))
        return self.output_net(torch.cat([last[0], last[1]], dim=-1))


class T2MEvaluator:
    """Loads the three T2M evaluator networks and exposes encode_motion / encode_text."""

    MAX_TEXT_LEN = 20
    DIM_WORD     = 300
    DIM_POS      = 15
    UNIT_LENGTH  = 4   # movement encoder downsampling factor

    def __init__(self, checkpoint_path, glove_dir, device):
        self.device = device

        self.movement_enc = MovementConvEncoder(259, 512, 512).to(device)
        self.motion_enc   = MotionEncoderBiGRUCo(512, 1024, 512, device).to(device)
        self.text_enc     = TextEncoderBiGRUCo(self.DIM_WORD, self.DIM_POS,
                                                512, 512, device).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.movement_enc.load_state_dict(ckpt['movement_encoder'])
        self.motion_enc.load_state_dict(ckpt['motion_encoder'])
        self.text_enc.load_state_dict(ckpt['text_encoder'])
        for m in (self.movement_enc, self.motion_enc, self.text_enc):
            m.eval()

        self.vectorizer = WordVectorizer(glove_dir)
        for required in ('sos', 'eos', 'unk'):
            if required not in self.vectorizer.word2vec:
                raise RuntimeError(f"GloVe vocab missing required token '{required}'")

    @torch.no_grad()
    def encode_motion(self, motions_eval_norm):
        """
        motions_eval_norm : iterable of (T_i, 263) float32 arrays, normalised with
                            the T2M evaluator's Mean/Std (NOT the HumanML3D ones).
        returns           : (N, 512) float32 array
        """
        results = []
        for m in motions_eval_norm:
            # Strip last 4 dims (foot contacts) — evaluator input is 259-dim.
            # Crop to a multiple of UNIT_LENGTH so conv downsampling is exact.
            T = (len(m) // self.UNIT_LENGTH) * self.UNIT_LENGTH
            x  = torch.from_numpy(m[:T, :-4]).unsqueeze(0).to(self.device)  # (1, T, 259)
            mv = self.movement_enc(x)                                        # (1, T//4, 512)
            l  = torch.tensor([mv.shape[1]], dtype=torch.long, device=self.device)
            results.append(self.motion_enc(mv, l)[0].cpu().numpy())
        return np.stack(results)  # (N, 512)

    @torch.no_grad()
    def encode_text(self, token_lists):
        """
        token_lists : list of lists of 'word/POS' strings (from HumanML3D text files,
                      second '#'-separated field).
        returns     : (N, 512) float32 array
        """
        N         = len(token_lists)
        fixed_len = self.MAX_TEXT_LEN + 2  # room for sos + eos
        word_embs = np.zeros((N, fixed_len, self.DIM_WORD), dtype=np.float32)
        pos_ohot  = np.zeros((N, fixed_len, self.DIM_POS),  dtype=np.float32)
        cap_lens  = np.zeros(N, dtype=np.int64)

        for i, tokens in enumerate(token_lists):
            tokens = list(tokens[:self.MAX_TEXT_LEN])
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            cap_lens[i] = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (fixed_len - len(tokens))
            for j, tok in enumerate(tokens):
                word_embs[i, j], pos_ohot[i, j] = self.vectorizer[tok]

        return self.text_enc(
            torch.from_numpy(word_embs).to(self.device),
            torch.from_numpy(pos_ohot).to(self.device),
            torch.from_numpy(cap_lens).to(self.device),
        ).cpu().numpy()  # (N, 512)
