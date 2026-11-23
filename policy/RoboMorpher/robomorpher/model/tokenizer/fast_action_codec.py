"""Frozen FAST+ whole-chunk codec, with strict decoding and explicit sequence IDs.

This uses the published FAST DCT/round/BPE algorithm, without executing remote
processor Python or fitting a new vocabulary. FAST tokens have no joint/time slot
meaning. The public physical output mask is separate from the sequence mask.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .action_codec import ActionCodec, ActionTokenSpec, register_action_codec


def _byte_alphabet():
    """Inverse of the GPT-2/ByteLevel reversible byte-to-Unicode alphabet."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {chr(char): byte for byte, char in zip(bs, cs)}


def _checkpoint_files(path, local_files_only):
    p = Path(path).expanduser()
    names = ("processor_config.json", "tokenizer.json")
    if p.is_dir():
        return {name: p / name for name in names}
    from huggingface_hub import hf_hub_download
    return {name: Path(hf_hub_download(str(path), name, local_files_only=local_files_only)) for name in names}


@register_action_codec("fast")
class FASTActionCodec(ActionCodec):
    """Encode full physical action chunks into variable-length FAST+ labels.

    Base BPE IDs are [0,V). EOS=V is predicted; BOS=V+1 and PAD=V+2 are
    embedding-only. ``max_tokens`` includes EOS. Oversize labels are errors,
    never truncated. Dataset q01/q99 statistics must be supplied explicitly.
    """
    supports_soft_angle_decode = False
    is_chunk_codec = True

    def __init__(self, tokenizer_path="physical-intelligence/fast", *,
                 stats_path=None, q01=None, q99=None, action_horizon=50,
                 action_dim=None, max_tokens=128, local_files_only=True,
                 num_arms=2, max_num_joints=7, action_representation="joint"):
        if action_representation != "joint":
            raise ValueError("FAST currently supports physical joint actions only")
        super().__init__(num_arms=num_arms, max_num_joints=max_num_joints)
        self.action_horizon = int(action_horizon)
        self.action_dim = self.num_arms * (self.max_num_joints + 1)
        self.max_tokens = int(max_tokens)
        if min(self.action_horizon, self.max_tokens) < 2:
            raise ValueError("FAST action_horizon/max_tokens must be >=2")
        if action_dim is not None and int(action_dim) != self.action_dim:
            raise ValueError("action_dim must equal arms * (max_num_joints + gripper)")
        if stats_path is not None:
            if q01 is not None or q99 is not None:
                raise ValueError("Use stats_path or explicit q01/q99, not both")
            stats = json.loads(Path(stats_path).expanduser().read_text())
            stats = stats.get("action", stats)
            q01, q99 = stats.get("q01"), stats.get("q99")
        lo = np.asarray(q01, dtype=np.float32)
        hi = np.asarray(q99, dtype=np.float32)
        if lo.shape != (self.action_dim,) or hi.shape != lo.shape:
            raise ValueError(f"FAST requires q01/q99 each shaped [{self.action_dim}]")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
            raise ValueError("FAST normalization statistics must be finite, q99 >= q01")
        # Constant coordinates are supported (with the same epsilon as openpi).
        self.register_buffer("q01", torch.from_numpy(lo.copy()), persistent=False)
        self.register_buffer("q99", torch.from_numpy(hi.copy()), persistent=False)
        self.normalization_epsilon = 1e-6
        files = _checkpoint_files(tokenizer_path, bool(local_files_only))
        config = json.loads(files["processor_config.json"].read_text())
        tokenizer_json = json.loads(files["tokenizer.json"].read_text())
        if tokenizer_json.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("FAST strict decoding requires the published ByteLevel tokenizer")
        from tokenizers import Tokenizer
        self.bpe_tokenizer = Tokenizer.from_file(str(files["tokenizer.json"]))
        self.vocab_size = self.bpe_tokenizer.get_vocab_size(with_added_tokens=True)
        if self.vocab_size != int(config["vocab_size"]):
            raise ValueError("FAST processor and BPE vocabulary sizes differ")
        if tokenizer_json.get("added_tokens"):
            raise ValueError("FAST vocabulary must not contain added special tokens")
        self.scale = float(config["scale"])
        self.min_token = int(config["min_token"])
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("FAST scale must be positive and finite")
        self.eos_id, self.bos_id, self.pad_id = self.vocab_size, self.vocab_size + 1, self.vocab_size + 2
        self.output_vocab_size, self.embedding_vocab_size = self.vocab_size + 1, self.vocab_size + 3
        inverse = _byte_alphabet()
        try:
            self._token_bytes = tuple(bytes(inverse[c] for c in self.bpe_tokenizer.id_to_token(i)) for i in range(self.vocab_size))
        except KeyError as exc:
            raise ValueError("Unsupported FAST ByteLevel vocabulary") from exc
        if any(not value for value in self._token_bytes):
            raise ValueError("FAST BPE contains an empty token")
        self._constraint_cache = {}
        hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}
        settings = dict(kind="fast", files=hashes, q01=lo.tolist(), q99=hi.tolist(),
                        horizon=self.action_horizon, action_dim=self.action_dim,
                        max_tokens=self.max_tokens, scale=self.scale, min_token=self.min_token,
                        normalization="q01_q99_clip_minus1_plus1", epsilon=self.normalization_epsilon,
                        padding="repeat_last_valid_action", gripper_decode="clip_0_1")
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        # Empty vocab_sizes is deliberate: a whole-chunk sequence has no frame slots.
        self.spec = ActionTokenSpec("fast", self.num_arms, self.max_num_joints, 0, (),
                                    "whole_chunk_variable_length_bpe_then_eos", digest)
        self.codec_metadata = dict(settings, fingerprint=digest, vocab_size=self.vocab_size,
                                   output_vocab_size=self.output_vocab_size, eos_id=self.eos_id,
                                   bos_id=self.bos_id, pad_id=self.pad_id,
                                   action_layout="arm_major_joint_angles_then_gripper",
                                   tokenizer_path=str(tokenizer_path), frozen=True)
        self.train(False)

    def _time_valid(self, examples):
        rows = [np.asarray(e.get("action_valid_mask", np.ones(self.action_horizon, dtype=bool))) for e in examples]
        values = np.stack(rows)
        if values.shape != (len(examples), self.action_horizon) or not np.isin(values, (0, 1)).all():
            raise ValueError("FAST action_valid_mask must be boolean [B,action_horizon]")
        tv = torch.as_tensor(values, device=self.device, dtype=torch.bool)
        if not bool(tv.any(-1).all()) or bool((~tv[:, :-1] & tv[:, 1:]).any()):
            raise ValueError("FAST valid action timesteps must form a nonempty prefix")
        return tv

    def physical_valid(self, examples):
        """Physical scalar validity [B,H,A*(J+1)], independent of BPE length."""
        _, m = self.geometry(examples)
        slots = torch.cat([m, m.any(-1, keepdim=True)], -1).flatten(1)
        return self._time_valid(examples)[..., None] & slots[:, None]

    @torch.no_grad()
    def encode_chunk(self, examples):
        from scipy.fft import dct
        _, m, q, g, _ = self._action_inputs(examples)
        if q.shape[1] != self.action_horizon:
            raise ValueError("FAST takes the original full action_horizon, without time compression")
        tv = self._time_valid(examples)
        physical = torch.cat([q, g[..., None]], -1).flatten(2)
        # Replicate the final observed action through dataset end padding.
        lengths = tv.sum(-1)
        last = physical[torch.arange(len(examples), device=self.device), lengths - 1]
        physical = torch.where(tv[..., None], physical, last[:, None])
        values = (2 * (physical - self.q01) / (self.q99 - self.q01 + self.normalization_epsilon) - 1).clamp(-1, 1)
        slots = torch.cat([m, m.any(-1, keepdim=True)], -1).flatten(1)
        values = values.masked_fill(~slots[:, None], 0)
        coefficients = np.around(dct(values.cpu().numpy(), axis=1, norm="ortho") * self.scale)
        ids = torch.full((len(examples), self.max_tokens), self.pad_id, device=self.device, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for b, coeff in enumerate(coefficients):
            chars = np.maximum(coeff.ravel() - self.min_token, 0).astype(np.int64)
            if np.any(chars > 0x10ffff) or np.any((chars >= 0xd800) & (chars <= 0xdfff)):
                raise ValueError("FAST coefficients are outside the Unicode scalar alphabet")
            text = "".join(map(chr, chars))
            row = self.bpe_tokenizer.encode(text, add_special_tokens=False).ids
            if len(row) + 1 > self.max_tokens:
                raise ValueError(f"FAST sample {b} needs {len(row)+1} tokens including EOS, capacity={self.max_tokens}; increase max_tokens (never truncate)")
            # Check the checkpoint's exact reversibility before using labels.
            if self._decode_text(row) != text:
                raise ValueError("FAST BPE changed the quantized coefficient sequence")
            labels = row + [self.eos_id]
            ids[b, :len(labels)] = torch.tensor(labels, device=self.device)
            valid[b, :len(labels)] = True
        return ids, valid

    encode_actions = encode_chunk

    def _decode_text(self, token_ids):
        try:
            return b"".join(self._token_bytes[i] for i in token_ids).decode("utf-8", errors="strict")
        except (IndexError, UnicodeDecodeError) as exc:
            raise ValueError("FAST sequence contains malformed UTF-8 coefficient bytes") from exc

    @torch.no_grad()
    def decode_chunk(self, ids, examples, valid_mask=None):
        from scipy.fft import idct
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("FAST requires integer sequence IDs [B,L]")
        if ids.shape[0] != len(examples) or not 1 <= ids.shape[1] <= self.max_tokens:
            raise ValueError("FAST sequence batch/length does not match codec capacity")
        valid = ids != self.pad_id
        if valid_mask is not None:
            if valid_mask.shape != ids.shape or valid_mask.dtype != torch.bool or not torch.equal(valid_mask.to(ids.device), valid):
                raise ValueError("FAST sequence mask must match non-PAD IDs exactly")
        if bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("FAST padding must follow the EOS, without gaps")
        rows = []
        target = self.action_horizon * self.action_dim
        for b, row in enumerate(ids.detach().cpu().tolist()):
            active = [i for i in row if i != self.pad_id]
            if not active or active[-1] != self.eos_id or active.count(self.eos_id) != 1:
                raise ValueError(f"FAST sample {b} requires exactly one terminal EOS")
            if any(i < 0 or i >= self.vocab_size for i in active[:-1]):
                raise ValueError(f"FAST sample {b} contains a non-BPE action ID")
            text = self._decode_text(active[:-1])
            if len(text) != target:
                raise ValueError(f"FAST sample {b} has {len(text)} DCT coefficients, expected {target}; refusing malformed action")
            coeff = np.fromiter(map(ord, text), dtype=np.float64, count=target).reshape(self.action_horizon, self.action_dim)
            values = idct((coeff + self.min_token) / self.scale, axis=0, norm="ortho")
            if not np.isfinite(values).all():
                raise ValueError("FAST decoded nonfinite actions")
            rows.append(values)
        values = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device)
        physical = (values + 1) / 2 * (self.q99 - self.q01 + self.normalization_epsilon) + self.q01
        if not bool(torch.isfinite(physical).all()):
            raise ValueError("FAST denormalized nonfinite actions")
        _, m = self.geometry(examples)
        physical_valid = self.physical_valid(examples)
        physical = physical.masked_fill(~physical_valid, 0).reshape(len(examples), self.action_horizon, self.num_arms, -1)
        q, g = physical[..., :-1], physical[..., -1].clamp(0, 1)
        result = self._result(q, g, m, physical_valid)
        result["action_token_mask"] = valid.to(self.device)
        result["action_token_ids"] = ids.to(self.device)
        return result

    decode_actions = decode_chunk

    def constrain_logits(self, prefix, logits):
        """Prevent malformed UTF-8, wrong coefficient count and premature EOS.

        Operates on actual ByteLevel bytes, so a BPE token may end in the middle
        of a Unicode coefficient. This checks syntax, not policy correctness.
        """
        if prefix.ndim != 2 or logits.shape != (len(prefix), self.output_vocab_size):
            raise ValueError("Invalid FAST generation prefix/logits shape")
        output = logits.clone()
        target = self.action_horizon * self.action_dim
        for b, row in enumerate(prefix.detach().cpu().tolist()):
            if self.eos_id in row:
                output[b, :self.vocab_size] = -torch.inf
                continue
            if any(i < 0 or i >= self.vocab_size for i in row):
                raise ValueError("Active FAST generation prefix contains a special/invalid ID")
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            try:
                text = decoder.decode(b"".join(self._token_bytes[i] for i in row), final=False)
            except UnicodeDecodeError as exc:
                raise ValueError("Malformed FAST prefix") from exc
            pending, _ = decoder.getstate()
            remaining = target - len(text)
            key = (pending, remaining)
            permitted = self._constraint_cache.get(key)
            if permitted is None:
                permitted = np.zeros(self.output_vocab_size, dtype=bool)
                for i, raw in enumerate(self._token_bytes):
                    candidate = codecs.getincrementaldecoder("utf-8")("strict")
                    candidate.setstate((pending, 0))
                    try:
                        extra = candidate.decode(raw, final=False)
                        trailing, _ = candidate.getstate()
                        permitted[i] = len(extra) <= remaining and (len(extra) < remaining or not trailing)
                    except UnicodeDecodeError:
                        pass
                permitted[self.eos_id] = remaining == 0 and not pending
                # Bound runtime cache when encountering long/outlier generation.
                if len(self._constraint_cache) > 4096:
                    self._constraint_cache.clear()
                self._constraint_cache[key] = permitted
            output[b].masked_fill_(~torch.as_tensor(permitted, device=logits.device), -torch.inf)
        return output
