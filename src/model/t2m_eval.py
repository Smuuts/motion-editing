"""
T2M evaluator models — architecture from Guo et al. (2022)
"Generating Diverse and Natural 3D Human Motions from Text."

Required files (see evaluate.py --help):
  data/t2m_evaluator/checkpoint/finest.tar
  data/t2m_evaluator/glove/our_vab_data.npy
  data/t2m_evaluator/glove/our_vab_words.pkl
  data/t2m_evaluator/glove/our_vab_idx.pkl
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


# POS tags and VIP word lists must match what T2M used during training.
POS_enumerator = {
    'VERB': 0, 'NOUN': 1, 'DET': 2, 'ADP': 3, 'NUM': 4, 'AUX': 5,
    'CONJ': 6, 'PRON': 7, 'ADJ': 8, 'ADV': 9,
    'Loc_VIP': 10, 'Body_VIP': 11, 'Obj_VIP': 12, 'Act_VIP': 13, 'OTHER': 14,
}

VIP_dict = {
    'Loc_VIP':  ['left', 'right', 'clockwise', 'counterclockwise',
                 'anticlockwise', 'forward', 'back', 'backward', 'up',
                 'down', 'straight', 'curve'],
    'Body_VIP': ['arm', 'arms', 'leg', 'legs', 'hand', 'hands', 'foot', 'feet',
                 'shoulder', 'shoulders', 'hip', 'hips', 'knee', 'knees',
                 'wrist', 'wrists', 'ankle', 'ankles', 'body', 'torso', 'head'],
    'Obj_VIP':  ['stair', 'stairs', 'ball', 'dumbbell', 'dumbbells'],
    'Act_VIP':  ['walk', 'walks', 'run', 'runs', 'jump', 'jumps', 'dance',
                 'dances', 'kick', 'kicks', 'punch', 'punches', 'throw',
                 'throws', 'catch', 'catches', 'pick', 'picks', 'put', 'puts',
                 'sit', 'sits', 'stand', 'stands', 'raise', 'raises', 'wave',
                 'waves', 'bend', 'bends', 'twist', 'twists', 'turn', 'turns',
                 'squat', 'squats', 'kneel', 'kneels', 'crawl', 'crawls',
                 'climb', 'climbs'],
}


class WordVectorizer:
    def __init__(self, glove_dir, prefix='our_vab'):
        vectors  = np.load(os.path.join(glove_dir, f'{prefix}_data.npy'))
        words    = pickle.load(open(os.path.join(glove_dir, f'{prefix}_words.pkl'), 'rb'))
        word2idx = pickle.load(open(os.path.join(glove_dir, f'{prefix}_idx.pkl'), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _pos_ohot(self, pos):
        vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        vec[POS_enumerator.get(pos, POS_enumerator['OTHER'])] = 1.0
        return vec

    def __call__(self, word, pos):
        vip_pos = next((k for k, vs in VIP_dict.items() if word in vs), None)
        effective_pos = vip_pos if vip_pos is not None else pos
        word_vec = self.word2vec.get(word, self.word2vec['unk']).astype(np.float32)
        return word_vec, self._pos_ohot(effective_pos)


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

        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"spacy not available: {exc}\n"
                "Install with:  pip install spacy && python -m spacy download en_core_web_sm"
            ) from exc

    @torch.no_grad()
    def encode_motion(self, motions_norm):
        """
        motions_norm : iterable of (T_i, 263) float32 arrays (normalised)
        returns      : (N, 512) float32 array
        """
        results = []
        for m in motions_norm:
            # Strip last 4 dims (foot contacts) — evaluator was trained on 259-dim input.
            x  = torch.from_numpy(m[:, :-4]).unsqueeze(0).to(self.device)  # (1, T, 259)
            mv = self.movement_enc(x)                                        # (1, T', 512)
            l  = torch.tensor([mv.shape[1]], dtype=torch.long, device=self.device)
            results.append(self.motion_enc(mv, l)[0].cpu().numpy())
        return np.stack(results)  # (N, 512)

    def _tokenize(self, text):
        tokens = []
        for tok in self.nlp(text.lower()):
            if not tok.is_space:
                tokens.append((tok.text, tok.pos_))
            if len(tokens) >= self.MAX_TEXT_LEN:
                break
        return tokens or [('unk', 'OTHER')]

    @torch.no_grad()
    def encode_text(self, texts):
        """
        texts   : list[str]
        returns : (N, 512) float32 array
        """
        N = len(texts)
        word_embs  = np.zeros((N, self.MAX_TEXT_LEN, self.DIM_WORD), dtype=np.float32)
        pos_onehot = np.zeros((N, self.MAX_TEXT_LEN, self.DIM_POS),  dtype=np.float32)
        cap_lens   = np.ones(N, dtype=np.int64)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            L = min(len(tokens), self.MAX_TEXT_LEN)
            cap_lens[i] = L
            for j, (word, pos) in enumerate(tokens[:L]):
                word_embs[i, j], pos_onehot[i, j] = self.vectorizer(word, pos)

        return self.text_enc(
            torch.from_numpy(word_embs).to(self.device),
            torch.from_numpy(pos_onehot).to(self.device),
            torch.from_numpy(cap_lens).to(self.device),
        ).cpu().numpy()  # (N, 512)
