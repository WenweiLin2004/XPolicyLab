"""Shared, pluggable action codecs for discrete VLA targets.

IDs are local categorical labels, never values in the VLM text vocabulary.
Only codecs define quantization, normalization, token layout and physical decoding.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib
import json
import math
from typing import Callable
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn
from robomorpher.utils.model.masking import binary_bool_mask
from .gripper_tokenizer import GripperTokenizer
from .utils.config import STEP_ARCHITECTURES

@dataclass(frozen=True)
class ActionTokenSpec:
    kind: str
    num_arms: int
    max_num_joints: int
    arm_token_count: int
    vocab_sizes: tuple[int, ...]
    layout: str
    fingerprint: str
    @property
    def tokens_per_frame(self): return len(self.vocab_sizes)
    def to_dict(self): return asdict(self)

_CODECS: dict[str, Callable] = {}
def register_action_codec(name):
    def wrap(factory):
        if name in _CODECS: raise ValueError(f"Duplicate action codec {name}")
        _CODECS[name] = factory
        return factory
    return wrap

def build_action_codec(config):
    cfg = dict(config)
    name = cfg.pop("type", "robomorpher")
    plugin = cfg.pop("plugin", None)
    if name == "fast" and plugin in (None, "models.tokenizer.fast_action_codec", "model.tokenizer.fast_action_codec"):
        plugin = "robomorpher.model.tokenizer.fast_action_codec"
    if plugin:
        importlib.import_module(plugin)  # Explicit config opt-in; plugin registers a factory.
    if name not in _CODECS:
        raise ValueError(f"Unknown codec {name!r}; registered={sorted(_CODECS)}")
    return _CODECS[name](**cfg)

def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()

class ActionCodec(nn.Module):
    """encode_actions / valid_tokens / decode_actions are the extension boundary."""
    spec: ActionTokenSpec
    supports_soft_angle_decode = False

    @property
    def urdf_feature_dim(self):
        """Optional frozen morphology encoder capability; zero means unavailable."""
        return 0

    def encode_urdf_condition(self, examples):
        raise ValueError("This action codec has no frozen URDF encoder")
    def __init__(self, num_arms=2, max_num_joints=7, gripper_bins=None, gripper=None,
                 action_representation="joint"):
        super().__init__()
        self.num_arms, self.max_num_joints = int(num_arms), int(max_num_joints)
        self.action_representation = str(action_representation).lower()
        if self.action_representation not in {"joint", "ee"}:
            raise ValueError("action_representation must be joint or ee")
        if min(self.num_arms, self.max_num_joints) < 1:
            raise ValueError("Positive arm/joint dimensions required")
        if gripper is not None and not isinstance(gripper, Mapping):
            raise TypeError("gripper must be a mapping with mode/num_bins/threshold")
        gripper_config = dict(gripper or {})
        # Preserve old configs, but never let an alias override an explicit policy.
        if gripper_bins is not None:
            if "num_bins" in gripper_config and gripper_config["num_bins"] != gripper_bins:
                raise ValueError("Conflicting gripper_bins and gripper.num_bins")
            gripper_config.setdefault("num_bins", gripper_bins)
        self.gripper_tokenizer = GripperTokenizer(**gripper_config)
        self.gripper_bins = self.gripper_tokenizer.vocab_size
        self.register_buffer("_device_anchor", torch.zeros(0), persistent=False)

    def _with_gripper_fingerprint(self, base_fingerprint):
        # uniform bins retain their old spec: vocab_sizes already bind K.
        # Binary decoding/threshold must also be bound even when K stays two.
        if self.gripper_tokenizer.mode == "uniform_bins":
            return base_fingerprint
        payload = {"arm_codec_fingerprint": base_fingerprint,
                   "gripper": self.gripper_tokenizer.metadata()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def device(self): return self._device_anchor.device

    @property
    def state_condition_dim(self):
        """Plugins may override together with state_condition()."""
        return self.num_arms * (8 * self.max_num_joints + 1)

    def train(self, mode=True):
        # A VLA must never update the tokenizer's neural weights, EMA or usage.
        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        if fn(torch.zeros(1, device=self.device)).dtype != torch.float32:
            raise ValueError("Keep action codec FP32; use autocast around the VLM/head, not model.bfloat16()")
        return super()._apply(fn, recurse)

    def _stack(self, examples, key, dtype=torch.float32):
        values = np.stack([e[key] for e in examples])
        if dtype == torch.bool and not np.isin(values, (0, 1)).all():
            raise ValueError(f"{key} must contain only boolean/0/1 mask values")
        return torch.as_tensor(values, device=self.device, dtype=dtype)

    def geometry(self, examples):
        u = self._stack(examples, "urdf_features")
        raw_m = self._stack(examples, "joint_valid_mask", torch.bool)
        expected = (len(examples), self.num_arms, self.max_num_joints)
        if raw_m.shape != expected or u.shape != (*expected, 6):
            raise ValueError(f"Expected mask {expected} and URDF xyz+rpy {(*expected,6)}")
        if not bool(raw_m.any(dim=(1,2)).all()): raise ValueError("Each sample needs at least one arm")
        # EE actions always have nine valid xyz+rot6d slots per active arm.
        # The shared joint_valid_mask still describes physical q padding.
        m = (raw_m.any(-1, keepdim=True).expand(-1, -1, self.max_num_joints)
             if self.action_representation == "ee" else raw_m)
        if not bool(torch.isfinite(u[raw_m]).all()): raise ValueError("Nonfinite valid URDF")
        return u.masked_fill(~m[...,None], 0), m

    def _action_inputs(self, examples):
        u,m = self.geometry(examples)
        q = self._stack(examples, "action_angle")
        g = self._stack(examples, "gripper")
        tv = self._stack(examples, "action_valid_mask", torch.bool)
        if q.ndim != 4 or q.shape[0] != len(examples) or q.shape[2:] != m.shape[1:]:
            raise ValueError("action_angle must be [T,A,J] per sample")
        if g.shape != q.shape[:3] or tv.shape != q.shape[:2]:
            raise ValueError("gripper [T,A] and action_valid_mask [T] required")
        valid = tv[:,:,None,None] & m[:,None]
        gv = tv[:,:,None] & m.any(-1)[:,None]
        if not bool(torch.isfinite(q[valid]).all()): raise ValueError("Nonfinite valid angle")
        if not bool(torch.isfinite(g[gv]).all()) or bool(((g[gv]<0)|(g[gv]>1)).any()):
            raise ValueError("Gripper must be physical normalized opening in [0,1], no silent clipping")
        return u,m,q.masked_fill(~valid,0),g.masked_fill(~gv,0),tv

    @torch.no_grad()
    def angle_targets(self, examples):
        """Physical-radian targets; padding is zero and excluded by the mask."""
        _, m, q, _, tv = self._action_inputs(examples)
        valid = tv[:, :, None, None] & m[:, None]
        return q.masked_fill(~valid, 0), valid

    def normalize_angle_auxiliary(self, angles):
        """Map physical angle targets to this codec's action training space."""
        return angles

    def angle_auxiliary_metadata(self):
        return {
            "units": "codec_action_normalized",
            "stats_source": "codec",
            "normalization_method": "identity",
        }

    def decode_soft_angles(self, logits, examples, valid_mask=None):
        """Optional training-only API. Plugins must opt in to this capability.

        Return physical angles and their boolean mask, both [B,T,A,J]. The
        codec stays frozen, but the returned angles must retain logits gradients.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support differentiable angle decoding")

    def _soft_angle_inputs(self, logits, examples, valid_mask):
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.ndim != 4:
            raise ValueError("Soft decode expects floating logits [B,T,S,Kmax]")
        b, t, slots, width = logits.shape
        if b != len(examples) or t < 1 or slots != self.spec.tokens_per_frame:
            raise ValueError("Soft-logit layout does not match codec")
        if logits.device != self.device or width < max(self.spec.vocab_sizes):
            raise ValueError("Soft logits need the codec device and complete slot vocabularies")
        u, m = self.geometry(examples)
        tv = self._stack(examples, "action_valid_mask", torch.bool)
        if tv.shape != (b, t):
            raise ValueError("Time mask has wrong shape")
        permitted = self.valid_tokens(examples, t, tv)
        v = permitted if valid_mask is None else binary_bool_mask(
            valid_mask, device=self.device, name="token_valid_mask")
        if v.shape != permitted.shape or bool((v & ~permitted).any()):
            raise ValueError("Soft decode cannot unmask padded joints, arms or timesteps")
        # Reuse the hard decoder's complete-active-arm contract, without lookup.
        safe_ids = torch.zeros_like(v, dtype=torch.long).masked_fill(~v, -1)
        self._validate_decode(safe_ids, examples, v)
        arm_valid = v.reshape(b, t, self.num_arms, -1)
        joint_valid = arm_valid.any(-1)[..., None] & m[:, None]
        arm_logits = logits.reshape(b, t, self.num_arms, -1, width)[..., :-1, :]
        return u, m, arm_logits, arm_valid[..., :-1], joint_valid

    @staticmethod
    def _soft_arm_probabilities(logits, token_valid, vocab_size):
        # Slice BEFORE softmax: NaN outside a slot's vocabulary or at padding
        # must neither poison outputs nor receive gradients. Gripper is absent.
        clean = logits[..., :vocab_size].float().masked_fill(~token_valid[..., None], 0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Nonfinite valid angle-code logits")
        return clean.softmax(-1).masked_fill(~token_valid[..., None], 0)

    def _gripper_ids(self, g):
        # _action_inputs already sanitizes absent arms and padded timesteps.
        return self.gripper_tokenizer.encode(g)

    def _decode_gripper(self, ids):
        # The public decode boundary has checked all valid IDs and -1 padding.
        return self.gripper_tokenizer.decode(ids, valid_mask=ids >= 0)

    def _joint_token_mask(self, m): raise NotImplementedError

    def valid_tokens(self, examples, horizon, action_valid_mask=None):
        _,m = self.geometry(examples)
        arm = torch.cat([self._joint_token_mask(m),m.any(-1,keepdim=True)],dim=-1)
        v = arm.reshape(len(examples),1,-1).expand(-1,horizon,-1)
        if action_valid_mask is not None:
            action_valid_mask = binary_bool_mask(action_valid_mask, device=self.device, name="action_valid_mask")
            if action_valid_mask.shape != (len(examples),horizon):
                raise ValueError("Time mask has wrong shape")
            v = v & action_valid_mask[...,None]
        return v

    def state_condition(self, examples, *, include_urdf=True, include_state=True):
        u,m = self.geometry(examples)
        # Geometry is independent of learned tokenizer injection; this is VLA context.
        if include_urdf: u = self.normalize_geometry(u,m)
        else: u = torch.zeros_like(u)
        if include_state:
            q = self._stack(examples,"state_angle")
            g = self._stack(examples,"state_gripper")
            state_m = self._stack(examples, "joint_valid_mask", torch.bool)
            if q.shape!=state_m.shape or g.shape!=state_m.shape[:2]: raise ValueError("Invalid observed state shape")
            if not bool(torch.isfinite(q[state_m]).all()) or not bool(torch.isfinite(g[state_m.any(-1)]).all()):
                raise ValueError("Nonfinite observed state")
            if bool(((g[state_m.any(-1)] < 0) | (g[state_m.any(-1)] > 1)).any()):
                raise ValueError("Observed gripper opening must be in [0,1]")
            q = q.masked_fill(~state_m, 0)
            # Joint state is radians. EE state is already xyz meters and rot6d.
            if self.action_representation != "ee":
                q = q / math.pi
            g = g.masked_fill(~state_m.any(-1),0)
        else:
            q = torch.zeros_like(m,dtype=u.dtype);g = torch.zeros_like(m[...,0],dtype=u.dtype)
        condition_mask = state_m if (include_state and self.action_representation == "ee") else m
        return torch.cat([u.flatten(2),condition_mask.float(),q,g[...,None]],dim=-1).flatten(1)

    def normalize_geometry(self,u,m):
        # Uniform-bin/custom codecs have no learned stats: SI xyz, rpy/pi.
        return torch.cat([u[...,:3],u[...,3:]/math.pi],-1).masked_fill(~m[...,None],0)

    def _result(self, q, g, m, token_valid):
        q = q.masked_fill(~m[:,None],0)
        g = g.masked_fill(~m.any(-1)[:,None],0)
        # Preserve actual arm DOFs for execution; never send padded joint commands.
        packed=[]
        for b in range(len(q)):
            arms=[]
            for a in range(self.num_arms):
                if bool(m[b,a].any()):
                    arms += [q[b,:,a,m[b,a]],g[b,:,a,None]]
            packed.append(torch.cat(arms,-1))
        angle_valid = (
            token_valid.reshape(token_valid.shape[0], token_valid.shape[1], self.num_arms, -1).any(-1)[..., None]
            & m[:, None]
        )
        result = {"action_angle":q,"gripper":g,"packed_actions":packed,
                  "joint_valid_mask":m,"token_valid_mask":token_valid,
                  "angle_valid_mask": angle_valid,
                  "action_representation": self.action_representation}
        if self.action_representation == "ee":
            result["action_ee"] = q
        return result

    def _validate_decode(self, ids, examples, valid_mask):
        if ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=3:
            raise ValueError("Expected integer IDs [B,T,S]")
        if ids.shape[0]!=len(examples) or ids.shape[-1]!=self.spec.tokens_per_frame:
            raise ValueError("Token layout does not match codec")
        v = self.valid_tokens(examples,ids.shape[1]) if valid_mask is None else valid_mask
        if v.shape!=ids.shape or v.dtype!=torch.bool: raise ValueError("Invalid token mask")
        structural=self.valid_tokens(examples,ids.shape[1])
        if bool((v & ~structural).any()):raise ValueError("Cannot unmask padded joints/arms")
        arm_v = v.reshape(*ids.shape[:2],self.num_arms,-1)
        arm_structural = structural.reshape_as(arm_v)
        active = arm_v.any(-1,keepdim=True)
        if bool(((arm_v != arm_structural) & active).any()):
            raise ValueError("Decode requires a complete set of tokens for each active arm")
        sizes = ids.new_tensor(self.spec.vocab_sizes)
        if bool(((ids[v]<0)).any()) or bool(((ids>=sizes)&v).any()):
            raise ValueError("Code outside position-specific vocabulary")
        if bool((ids[~v]!=-1).any()): raise ValueError("All padding IDs must equal -1")
        return v

@register_action_codec("uniform_bins")
class UniformBinCodec(ActionCodec):
    """Baseline: bin each physical joint independently; no learned checkpoint."""
    supports_soft_angle_decode = True

    def __init__(self, num_bins=256, angle_low=-2.0 * math.pi, angle_high=2.0 * math.pi,
                 angle_wrap="none", checkpoint=None, value_low=None, value_high=None,
                 action_representation="joint", **kwargs):
        if checkpoint is not None:
            raise ValueError("uniform_bins does not use a learned checkpoint; set checkpoint=null")
        if str(action_representation).lower() != "joint":
            raise ValueError("uniform_bins only supports joint action_representation")
        super().__init__(action_representation=action_representation, **kwargs)
        self.num_bins=int(num_bins);self.angle_low=float(angle_low);self.angle_high=float(angle_high)
        if self.action_representation == "ee":
            default_lo = [-2.0, -2.0, -2.0] + [-1.0] * 6
            default_hi = [2.0, 2.0, 2.0] + [1.0] * 6
            lo = np.asarray(default_lo if value_low is None else value_low, dtype=np.float32)
            hi = np.asarray(default_hi if value_high is None else value_high, dtype=np.float32)
            if lo.shape != (self.max_num_joints,) or hi.shape != (self.max_num_joints,):
                raise ValueError("EE value_low/value_high must have shape [9]")
            if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
                raise ValueError("EE bounds must be finite with high > low")
            self.value_low, self.value_high = lo, hi
        else:
            self.value_low = np.full((self.max_num_joints,), self.angle_low, dtype=np.float32)
            self.value_high = np.full((self.max_num_joints,), self.angle_high, dtype=np.float32)
        self.joint_angle_wrap = str(angle_wrap)
        if self.joint_angle_wrap != "none":
            raise ValueError("uniform_bins only supports raw angle_wrap=none")
        if (self.num_bins<2 or not math.isfinite(self.angle_low) or not math.isfinite(self.angle_high)
                or not self.angle_low<self.angle_high):raise ValueError("Invalid binning bounds")
        # Joint checkpoints hash one scalar bound pair. Per-joint float32 lists
        # round 卤2蟺 and no longer match those fingerprints. EE bounds differ by
        # axis, so they stay as lists.
        if self.action_representation == "ee":
            hash_low, hash_high = self.value_low.tolist(), self.value_high.tolist()
        else:
            hash_low, hash_high = float(self.angle_low), float(self.angle_high)
        settings=dict(kind="uniform_bins",num_bins=self.num_bins,low=hash_low,high=hash_high,
                      angle_wrap=self.joint_angle_wrap,
                      arms=self.num_arms,joints=self.max_num_joints,gripper_bins=self.gripper_bins)
        self.codec_metadata = dict(settings, angle_units="radians", angle_wrap=self.joint_angle_wrap,
                                   angle_bounds=[self.angle_low,self.angle_high],
                                   angle_representative="bin_center", gripper_bounds=[0.0,1.0],
                                   gripper_representative="bin_center",
                                   action_representation=("per_arm_ee9_then_gripper" if self.action_representation == "ee"
                                                          else "per_joint_angle_then_gripper"),
                                   quantization="floor_then_clip_inclusive_upper_endpoint")
        self.codec_metadata["gripper_representative"] = self.gripper_tokenizer.metadata()["representative"]
        self.codec_metadata["gripper_tokenization"] = self.gripper_tokenizer.metadata()
        digest=self._with_gripper_fingerprint(hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest())
        vocab=tuple(([self.num_bins]*self.max_num_joints+[self.gripper_bins])*self.num_arms)
        self.spec=ActionTokenSpec("uniform_bins",self.num_arms,self.max_num_joints,
                                 self.max_num_joints,vocab,"arm_major_joint_bins_then_gripper",digest)
        self.train(False)

    def _joint_token_mask(self,m):return m

    @torch.no_grad()
    def angle_targets(self, examples):
        q, valid = super().angle_targets(examples)
        lo = torch.as_tensor(self.value_low, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        hi = torch.as_tensor(self.value_high, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        # Match verified simulator joint export roundoff; EE bounds stay strict.
        tolerance = 1e-3 if self.action_representation == "joint" else 1e-6
        if bool(((q < lo - tolerance) | (q > hi + tolerance))[valid].any()):
            raise ValueError("Angle outside configured bins")
        return q, valid

    def decode_soft_angles(self, logits, examples, valid_mask=None):
        with torch.autocast(device_type=self.device.type, enabled=False):
            _, _, arm_logits, token_valid, joint_valid = self._soft_angle_inputs(logits, examples, valid_mask)
            p = self._soft_arm_probabilities(arm_logits, token_valid, self.num_bins)
            if self.action_representation == "ee":
                lo = torch.as_tensor(self.value_low, device=self.device)
                hi = torch.as_tensor(self.value_high, device=self.device)
                centers = lo[:, None] + (torch.arange(self.num_bins, device=self.device).float()[None, :] + .5) / self.num_bins * (hi - lo)[:, None]
                return torch.einsum("btajk,jk->btaj", p, centers).masked_fill(~joint_valid, 0), joint_valid
            centers = self.angle_low + (torch.arange(self.num_bins, device=self.device).float() + .5) / self.num_bins * (self.angle_high - self.angle_low)
            return (p @ centers).masked_fill(~joint_valid, 0), joint_valid

    def normalize_angle_auxiliary(self, angles):
        # Uniform bins use their fixed tokenizer range as the normalized space.
        return (2 * (angles - self.angle_low) / (self.angle_high - self.angle_low) - 1).clamp(-1, 1)

    def angle_auxiliary_metadata(self):
        return {
            "units": "codec_action_normalized",
            "stats_source": "uniform_bin_bounds",
            "normalization_method": "fixed_range",
            "angle_bounds": [self.angle_low, self.angle_high],
        }

    @torch.no_grad()
    def encode_actions(self,examples):
        _,m,q,g,tv=self._action_inputs(examples)
        valid=tv[:,:,None,None]&m[:,None]
        lo = torch.as_tensor(self.value_low, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        hi = torch.as_tensor(self.value_high, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        tolerance = 1e-3 if self.action_representation == "joint" else 1e-6
        if bool(((q<lo-tolerance)|(q>hi+tolerance))[valid].any()):
            raise ValueError("Action value outside configured bins")
        ids=((q-lo)/(hi-lo)*self.num_bins).floor().long().clamp(0,self.num_bins-1)
        ids=torch.cat([ids,self._gripper_ids(g)[...,None]],-1).flatten(2)
        vm=self.valid_tokens(examples,q.shape[1],tv)
        return ids.masked_fill(~vm,-1),vm

    @torch.no_grad()
    def decode_actions(self,ids,examples,valid_mask=None):
        v=self._validate_decode(ids,examples,valid_mask);_,m=self.geometry(examples)
        x=ids.reshape(*ids.shape[:2],self.num_arms,self.max_num_joints+1)
        lo = torch.as_tensor(self.value_low, device=x.device, dtype=torch.float32).reshape(1, 1, 1, -1)
        hi = torch.as_tensor(self.value_high, device=x.device, dtype=torch.float32).reshape(1, 1, 1, -1)
        q=lo+(x[...,:-1].float()+.5)/self.num_bins*(hi-lo)
        g=self._decode_gripper(x[...,-1])
        frame=v.reshape(*ids.shape[:2],self.num_arms,-1).any(-1)
        return self._result(q.masked_fill(~frame[...,None],0),g.masked_fill(~frame,0),m,v)

@register_action_codec("robomorpher")
class RoboMorpherActionCodec(ActionCodec):
    """Frozen tokenizer checkpoint is the single source of architecture and stats."""
    supports_soft_angle_decode = True

    def __init__(self,checkpoint,num_arms=2,gripper_bins=None,max_num_joints=None,gripper=None,
                 action_representation="joint"):
        if str(action_representation).lower() != "joint":
            raise ValueError("RoboMorpherActionCodec only supports joint action_representation")
        from robomorpher.model.tokenizer import build_tokenizer_from_config
        from robomorpher.model.tokenizer.utils.config import normalize_model_config
        from robomorpher.utils.data.lazy_data.normalization import coerce_norm_stats, validate_norm_stats
        path=Path(checkpoint).expanduser().resolve(strict=True)
        ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        cfg=normalize_model_config(ck["model_config"])
        j=int(cfg["max_num_joints"])
        if max_num_joints is not None and int(max_num_joints)!=j:
            raise ValueError("max_num_joints must match tokenizer checkpoint")
        super().__init__(num_arms=num_arms,max_num_joints=j,gripper_bins=gripper_bins,gripper=gripper)
        self.model=build_tokenizer_from_config(ck["model_config"],state_dict=ck["model_state_dict"])
        self.model.load_state_dict(ck["model_state_dict"],strict=True)
        self.model.requires_grad_(False);self.model.eval()
        self.model_config=ck["model_config"];self.config=cfg
        self.norm=coerce_norm_stats(ck["norm"],normalization_method=ck["norm"]["normalization_method"])
        self.normalization_method=self.norm["normalization_method"]
        # Frozen statistics bind the raw physical-angle contract. Metadata-free
        # raw checkpoints remain readable, but no principal-angle fallback exists.
        self.joint_angle_wrap = self.norm.get("metadata", {}).get("angle_wrap", "none")
        if self.joint_angle_wrap != "none":
            raise ValueError(
                f"Tokenizer checkpoint uses removed angle convention {self.joint_angle_wrap!r}; "
                "recompute it with joint_angle_wrap=none"
            )
        validate_norm_stats(self.norm, max_num_joints=j,
                            urdf_rotation_6d=cfg["urdf_rotation_6d"],
                            normalization_method=self.normalization_method)
        for k,v in self.norm.items():
            if k.startswith(("x_","y_")) and isinstance(v,(list,tuple,np.ndarray,torch.Tensor)):
                self.register_buffer("norm_"+k,torch.as_tensor(v,dtype=torch.float32),persistent=False)
        self.checkpoint=str(path)
        self.is_frame=cfg["architecture"] in STEP_ARCHITECTURES
        self.groups=int(cfg["rvq_num_quantizers"] if self.is_frame else cfg["num_codebooks"])
        self.latents=int(cfg["frame_num_latents"] if self.is_frame else j)
        self.k=int(cfg["codebook_size"] if self.is_frame else cfg["codebook_size"]//self.groups)
        n=self.latents*self.groups
        self.spec=ActionTokenSpec("robomorpher",self.num_arms,j,n,
            tuple(([self.k]*n+[self.gripper_bins])*self.num_arms),
            "arm_major_latent_slot_major_codebook_fast_then_gripper",self._with_gripper_fingerprint(_hash_file(path)))
        self.tokenizer_metadata=self.model.quantizer_metadata()
        self.codec_metadata=dict(self.tokenizer_metadata,kind="robomorpher", checkpoint=self.checkpoint,
                                 gripper_bins=self.gripper_bins,gripper_bounds=[0.0,1.0],
                                 gripper_representative=self.gripper_tokenizer.metadata()["representative"],
                                 gripper_tokenization=self.gripper_tokenizer.metadata())
        self.codec_metadata["angle_wrap"] = "none"
        if getattr(self.model.quantizer,"has_pending_ema",False):raise ValueError("Pending tokenizer EMA")
        self.train(False)
        del ck

    def copy_decoder_adaln(self):
        """Return a frozen compatibility copy; tokenizer decoder AdaLN never trains in policy stage."""
        import copy
        if not self.is_frame or not getattr(self.model, "urdf_conditioning", False):
            raise ValueError("Decoder AdaLN adaptation requires a conditioned tokenizer decoder")
        layers = self.model.post_quant_proj
        if not layers or any(not hasattr(layer, "adaLN_modulation") for layer in layers):
            raise ValueError("Every tokenizer decoder layer must expose AdaLN")
        return nn.ModuleList([
            copy.deepcopy(layer.adaLN_modulation).requires_grad_(False) for layer in layers
        ])

    def decode_adapted_actions(self, ids, examples, valid_mask, decoder_adaln):
        """Hard code lookup plus original decoder, with policy-owned AdaLN weights.

        No soft codes or straight-through estimator. Gradients reach only the
        supplied AdaLN modules; source encoder, EMA codebooks and stats never change.
        """
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("The source codec must remain frozen and in eval")
        if not self.is_frame or not self.model.urdf_conditioning:
            raise ValueError("Adapted hard decoding requires a URDF-conditioned tokenizer")
        if len(decoder_adaln) != len(self.model.post_quant_proj):
            raise ValueError("Decoder AdaLN count differs from source decoder")
        v = self._validate_decode(ids, examples, valid_mask)
        u, m = self.geometry(examples)
        b, t, _ = ids.shape
        a, j = self.num_arms, self.max_num_joints
        shaped = ids.reshape(b, t, a, -1)
        active = v.reshape(b, t, a, -1).any(-1).reshape(-1)
        fm = m[:, None].expand(-1, t, -1, -1).reshape(-1, j)
        codes = shaped[..., :-1].reshape(-1, self.latents, self.groups)
        # Encode morphology once per batch/arm, not once per action frame.
        features, _ = self.encode_urdf_condition(examples)
        features = features.reshape(b, a, j, self.urdf_feature_dim)
        cond_all = features[:, None].expand(-1, t, -1, -1, -1).reshape(
            b*t*a, j, self.urdf_feature_dim)
        modules = [
            module(cond_all.to(next(module.parameters()).dtype)).float()[active]
            for module in decoder_adaln
        ]
        q = u.new_zeros((b*t*a, j))
        ep_full = None
        if bool(active.any()):
            with torch.no_grad():
                z = self.model.quantizer.idx_to_f(
                    codes[active], mask=self.model._latent_mask(fm[active]))
            condition = features[:, None].expand(-1, t, -1, -1, -1).reshape(
                b*t*a, j, self.urdf_feature_dim)[active]
            output = self.model._decode(
                z, fm[active], urdf_condition=condition, decoder_modulations=modules)
            ep, qp = self.model._split_output(output)
            index = active.nonzero(as_tuple=False).flatten()
            q = q.index_copy(0, index, self._denormalize_y(qp, slice(9, 10))[..., 0])
            if ep.shape[-1]:
                ep_full = u.new_zeros((b*t*a, j, 9)).index_copy(
                    0, index, self._denormalize_y(ep, slice(0, 9)))
        grip = self._decode_gripper(shaped[..., -1]).masked_fill(~active.reshape(b,t,a), 0)
        result = self._result(q.reshape(b,t,a,j), grip, m, v)
        if ep_full is not None:
            result["direct_ee"] = ep_full.reshape(b,t,a,j,9).masked_fill(~m[:,None,...,None],0)
        result["action_decode_mode"] = "hard_decoder_adaln"
        return result

    def _stat(self,key):return getattr(self,"norm_"+key)

    def _normalize(self,x,prefix):
        if self.normalization_method=="zscore":
            return (x-self._stat(prefix+"_mean"))/self._stat(prefix+"_std")
        lo,hi=("min","max") if self.normalization_method=="minmax" else ("q01","q99")
        y=(x-self._stat(prefix+"_"+lo))/(self._stat(prefix+"_"+hi)-self._stat(prefix+"_"+lo)+1e-6)*2-1
        return y if self.normalization_method=="minmax" else y.clamp(-1,1)

    def _denormalize_y(self,x,sl):
        if self.normalization_method=="zscore":
            return x*self._stat("y_std")[...,sl]+self._stat("y_mean")[...,sl]
        lo,hi=("min","max") if self.normalization_method=="minmax" else ("q01","q99")
        return (x+1)/2*(self._stat("y_"+hi)[...,sl]-self._stat("y_"+lo)[...,sl]+1e-6)+self._stat("y_"+lo)[...,sl]

    def normalize_angle_auxiliary(self, angles):
        """Normalize physical radians with the frozen tokenizer y-angle stats.

        This is the action coordinate system used by the tokenizer decoder.
        It supports both shared-feature [D] and reference/per-slot [J,D] stats.
        """
        if not isinstance(angles, torch.Tensor) or not angles.is_floating_point():
            raise ValueError("angles must be a floating tensor")
        def stat(name):
            value = self._stat("y_" + name)[..., 9]
            if value.ndim:
                value = value.reshape((1,) * (angles.ndim - 1) + (value.shape[-1],))
            return value
        if self.normalization_method == "zscore":
            return (angles - stat("mean")) / stat("std")
        lo, hi = ("min", "max") if self.normalization_method == "minmax" else ("q01", "q99")
        normalized = (angles - stat(lo)) / (stat(hi) - stat(lo) + 1e-6) * 2 - 1
        return normalized.clamp(-1, 1) if self.normalization_method == "q99" else normalized

    def angle_auxiliary_metadata(self):
        return {
            "units": "tokenizer_y_normalized",
            "stats_source": "eat_tokenizer_checkpoint.norm.y_angle",
            "normalization_method": self.normalization_method,
            "normalization_scope": self.norm.get("metadata", {}).get("scope", "per_slot"),
            "tokenizer_checkpoint": self.checkpoint,
        }

    def _input_features(self,u,q):
        if self.config["urdf_rotation_6d"]:
            from robomorpher.utils.geometry.diff_fk import euler_xyz_to_matrix,matrix_to_rotation_6d
            u=torch.cat([u[...,:3],matrix_to_rotation_6d(euler_xyz_to_matrix(u[...,3:]))],-1)
        return torch.cat([u,q[...,None]],-1)

    @property
    def urdf_feature_dim(self):
        if (not self.is_frame or not getattr(self.model, "urdf_injection_enabled", False)
                or not getattr(self.model, "use_urdf_split", False)):
            return 0
        return int(self.model.n_latent_dims)

    @torch.no_grad()
    def encode_urdf_condition(self, examples):
        """Original frozen URDF stream, no angle/EE encoder or quantizer call.

        Keep arm/joint slots instead of flattening all geometry into one token.
        Empty arms are not sent through an all-masked attention sequence.
        """
        if not self.urdf_feature_dim:
            raise ValueError("Tokenizer checkpoint has no enabled URDF encoder; use raw_features conditioning")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("URDF condition extraction requires a frozen eval tokenizer")
        with torch.autocast(device_type=self.device.type, enabled=False):
            u, m = self.geometry(examples)
            b, a, j = m.shape
            normalized = self._normalize(self._input_features(u, torch.zeros_like(m, dtype=u.dtype)), "x")
            normalized = normalized.masked_fill(~m[..., None], 0)
            urdf = normalized[..., :self.model.urdf_dim].reshape(b*a, j, self.model.urdf_dim)
            mask = m.reshape(b*a, j)
            active = mask.any(-1)
            features = u.new_zeros((b*a, j, self.urdf_feature_dim))
            if bool(active.any()):
                features[active] = self.model.encode_urdf(urdf[active], mask[active])
            features = features.masked_fill(~mask[..., None], 0)
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError("Nonfinite frozen URDF encoder features")
            return features.reshape(b, a*j, self.urdf_feature_dim), m.reshape(b, a*j)

    def normalize_geometry(self,u,m):
        x=self._input_features(u,torch.zeros_like(m,dtype=u.dtype))
        nx=self._normalize(x,"x").masked_fill(~m[...,None],0)
        # VLA condition remains xyz+rpy even for historical rot6d-input checkpoints.
        return nx[...,:6] if not self.config["urdf_rotation_6d"] else super().normalize_geometry(u,m)

    def _joint_token_mask(self,m):
        return m[...,None].expand(*m.shape,self.groups).flatten(-2)

    def decode_soft_angles(self, logits, examples, valid_mask=None, decoder_adaln=None):
        """Expected frozen code vectors -> original decoder -> physical radians.

        RVQ sums full-width stage vectors; PQ concatenates subspaces. This path
        never calls a quantizer forward, initializes codes, or updates EMA/usage.
        It intentionally does not change decode_actions' hard-ID execution path.
        """
        if self.model.training:
            raise RuntimeError("Soft action decoding requires a frozen eval tokenizer")
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Soft action decoding requires frozen tokenizer parameters")
        if decoder_adaln is not None:
            if not self.model.urdf_conditioning:
                raise ValueError("Soft decoder AdaLN requires URDF-conditioned tokenizer")
            if len(decoder_adaln) != len(self.model.post_quant_proj):
                raise ValueError("Decoder AdaLN count differs from source decoder")
        quantizer = self.model.quantizer
        if hasattr(quantizer, "_require_initialized"):
            quantizer._require_initialized()
        with torch.autocast(device_type=self.device.type, enabled=False):
            u, m, arm_logits, token_valid, joint_valid = self._soft_angle_inputs(logits, examples, valid_mask)
            p = self._soft_arm_probabilities(arm_logits, token_valid, self.k)
            b, t = logits.shape[:2]
            j = self.max_num_joints
            active = joint_valid.any(-1).reshape(-1)
            fm = m[:, None].expand(-1, t, -1, -1).reshape(-1, j)
            fu = u[:, None].expand(-1, t, -1, -1, -1).reshape(-1, j, 6)
            probs = p.reshape(-1, self.latents, self.groups, self.k)
            # Retain a finite, zero-gradient connection for all-padding batches.
            q = p.sum() * 0 + torch.zeros((b * t * self.num_arms, j), device=self.device)
            if bool(active.any()):
                values = []
                if not self.is_frame or self.model.quantizer_type != "rvq":
                    raise ValueError("Soft angle decoding requires the frame residual VQ tokenizer")
                for group_index, group in enumerate(quantizer.codebooks):
                    table = group.codebook.detach().float()
                    if table.ndim != 2 or table.shape[0] != self.k or not bool(torch.isfinite(table).all()):
                        raise ValueError("Invalid frozen codebook for soft angle decoding")
                    values.append(probs[active, :, group_index] @ table)
                z = torch.zeros_like(values[0])
                for value in values:
                    z = z + value
                if decoder_adaln is not None:
                    # The source tokenizer and URDF encoder remain frozen.
                    # Soft code expectations and policy-owned AdaLN copies
                    # stay in the autograd graph.
                    encoded, _ = self.encode_urdf_condition(examples)
                    encoded = encoded.reshape(
                        b, self.num_arms, j, self.urdf_feature_dim
                    )
                    cond_all = encoded[:, None].expand(
                        -1, t, -1, -1, -1
                    ).reshape(
                        b * t * self.num_arms, j, self.urdf_feature_dim
                    )
                    condition = cond_all[active]
                    modulations = [
                        module(cond_all.to(
                            dtype=next(module.parameters()).dtype
                        )).float()[active]
                        for module in decoder_adaln
                    ]
                    _, qp = self.model._split_output(self.model._decode(
                        z, fm[active],
                        urdf_condition=condition,
                        decoder_modulations=modulations,
                    ))
                else:
                    cond = {}
                    if self.model.urdf_conditioning:
                        normalized = self._normalize(
                            self._input_features(
                                fu[active], torch.zeros_like(q[active])
                            ), "x"
                        )
                        normalized = normalized.masked_fill(
                            ~fm[active, ..., None], 0
                        )
                        cond = {"urdf_features": normalized[..., :6]}
                    _, qp = self.model._split_output(
                        self.model._decode(z, fm[active], **cond)
                    )
                q[active] = self._denormalize_y(qp, slice(9, 10))[..., 0]
            return q.reshape(b, t, self.num_arms, j).masked_fill(~joint_valid, 0), joint_valid

    @torch.no_grad()
    def encode_normalized(self,x,mask,ee=None):
        """Exact prepared-data interface for compatibility regression and caches."""
        self.model.eval()
        kwargs={"ee_features":ee} if self.config.get("input_with_ee",False) else {}
        with torch.autocast(device_type=x.device.type,enabled=False):
            ids,_=self.model.action_to_idx(x.float(),mask,**kwargs)
        if not self.is_frame:
            raise ValueError("Action encoding requires the frame residual VQ tokenizer")
        return ids.reshape(len(x),self.spec.arm_token_count)

    @torch.no_grad()
    def encode_actions(self,examples):
        from robomorpher.utils.geometry.diff_fk import forward_kinematics_6d
        u,m,q,g,tv=self._action_inputs(examples);b,t,a,j=q.shape
        fm=(tv[:,:,None,None]&m[:,None]).reshape(-1,j)
        fu=u[:,None].expand(-1,t,-1,-1,-1).reshape(-1,j,6)
        fq=q.reshape(-1,j)
        good=fm.any(-1)
        out=torch.full((b*t*a,self.spec.arm_token_count),-1,dtype=torch.long,device=self.device)
        if bool(good.any()):
            uu,qq,mm=fu[good],fq[good],fm[good]
            # Raw physical radians are part of the tokenizer checkpoint contract.
            xx=self._input_features(uu,qq)
            nx=self._normalize(xx,"x").masked_fill(~mm[...,None],0)
            ee=None
            if self.config.get("input_with_ee",False):
                # Ground-truth action FK is used ONLY to construct training labels.
                # Inference never needs future angles/EE.
                yy=torch.cat([forward_kinematics_6d(uu,qq,mm),qq[...,None]],-1)
                ee=self._normalize(yy,"y")[...,:9].masked_fill(~mm[...,None],0)
            out[good]=self.encode_normalized(nx,mm,ee)
        ids=out.reshape(b,t,a,-1)
        ids=torch.cat([ids,self._gripper_ids(g)[...,None]],-1).flatten(2)
        vm=self.valid_tokens(examples,t,tv)
        return ids.masked_fill(~vm,-1),vm

    @torch.no_grad()
    def decode_actions(self,ids,examples,valid_mask=None):
        v=self._validate_decode(ids,examples,valid_mask);u,m=self.geometry(examples)
        b,t,_=ids.shape;a=self.num_arms;j=self.max_num_joints
        shaped=ids.reshape(b,t,a,-1)
        fv=v.reshape(b,t,a,-1).any(-1).reshape(-1)
        fm=m[:,None].expand(-1,t,-1,-1).reshape(-1,j)
        fu=u[:,None].expand(-1,t,-1,-1,-1).reshape(-1,j,6)
        codes=shaped[...,:-1].reshape(-1,self.spec.arm_token_count)
        q=torch.zeros((b*t*a,j),device=self.device);ee=None
        if bool(fv.any()):
            normalized=self._normalize(self._input_features(fu[fv],torch.zeros_like(q[fv])),"x")
            normalized=normalized.masked_fill(~fm[fv,...,None],0)
            with torch.autocast(device_type=self.device.type,enabled=False):
                if self.is_frame:
                    cond = {"urdf_features": normalized[...,:6]} if self.model.urdf_conditioning else {}
                    ep,qp=self.model.decode_ids(codes[fv],fm[fv],**cond)
                else:
                    raise ValueError("Action decoding requires the frame residual VQ tokenizer")
            q[fv]=self._denormalize_y(qp,slice(9,10))[...,0]
            if ep.shape[-1]:
                ee=torch.zeros((b*t*a,j,9),device=self.device)
                ee[fv]=self._denormalize_y(ep,slice(0,9))
        grip=self._decode_gripper(shaped[...,-1]).masked_fill(~fv.reshape(b,t,a),0)
        result=self._result(q.reshape(b,t,a,j),grip,m,v)
        if ee is not None:result["direct_ee"]=ee.reshape(b,t,a,j,9).masked_fill(~m[:,None,...,None],0)
        return result

class ContinuousActionCodec(ActionCodec):
    """Physical action layout/normalizer used when the tokenizer is disabled.

    It deliberately has no quantizer, codebook, tokenizer checkpoint, or token
    IDs. The inherited geometry/mask helpers are retained so the policy and
    dataset keep one bimanual layout contract. ``q99`` follows the StarVLA
    q01/q99 affine transform. ``minmax`` uses the same slot layout with the
    dataset minimum and maximum.
    """
    joint_angle_wrap = "none"

    def __init__(self, num_arms=2, max_num_joints=7,
                 angle_low=-2 * math.pi, angle_high=2 * math.pi,
                 normalization="q99", stats_path=None, stats=None, tokenizer_stats_path=None,
                 value_low=None, value_high=None, action_representation="joint"):
        super().__init__(num_arms=num_arms, max_num_joints=max_num_joints, action_representation=action_representation)
        self.angle_low, self.angle_high = float(angle_low), float(angle_high)
        if self.action_representation == "ee":
            lo = np.asarray([-2.0, -2.0, -2.0] + [-1.0] * 6 if value_low is None else value_low, dtype=np.float32)
            hi = np.asarray([2.0, 2.0, 2.0] + [1.0] * 6 if value_high is None else value_high, dtype=np.float32)
            if lo.shape != (self.max_num_joints,) or hi.shape != (self.max_num_joints,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
                raise ValueError("EE value_low/value_high must have shape [9] and high > low")
            self.value_low, self.value_high = lo, hi
        else:
            self.value_low = np.full((self.max_num_joints,), self.angle_low, dtype=np.float32)
            self.value_high = np.full((self.max_num_joints,), self.angle_high, dtype=np.float32)
        if not math.isfinite(self.angle_low) or not math.isfinite(self.angle_high) or self.angle_low >= self.angle_high:
            raise ValueError("Invalid continuous angle bounds")
        self.action_dim = self.num_arms * (self.max_num_joints + 1)
        self.normalization = str(normalization)
        if self.normalization not in {"q99", "tokenizer_stats", "minmax"}:
            raise ValueError("continuous normalization must be q99, tokenizer_stats, or minmax")
        self.stats_path = str(stats_path) if stats_path else None
        self.tokenizer_stats_path = str(tokenizer_stats_path) if tokenizer_stats_path else None
        self._tokenizer_norm_method = None
        q01 = q99 = None
        token_lo = token_hi = token_mean = token_std = None
        if self.normalization == "tokenizer_stats":
            if not self.tokenizer_stats_path:
                raise ValueError("tokenizer_stats normalization requires tokenizer_stats_path")
            from robomorpher.utils.data.lazy_data.normalization import coerce_norm_stats
            raw_checkpoint = torch.load(self.tokenizer_stats_path, map_location="cpu", weights_only=False, mmap=True)
            if not isinstance(raw_checkpoint, Mapping) or "norm" not in raw_checkpoint:
                raise ValueError("tokenizer_stats_path must be a tokenizer checkpoint containing norm")
            norm = coerce_norm_stats(raw_checkpoint["norm"], normalization_method=raw_checkpoint["norm"].get("normalization_method"))
            method = str(norm.get("normalization_method", "minmax"))
            if method not in {"minmax", "q99", "zscore"}:
                raise ValueError(f"Unsupported tokenizer normalization method: {method}")
            self._tokenizer_norm_method = method
            def angle_slots(key, default):
                if key not in norm:
                    return np.full((self.max_num_joints,), float(default), dtype=np.float32)
                arr = np.asarray(norm[key], dtype=np.float32)
                if arr.ndim == 1:
                    value = float(arr[9])
                    return np.full((self.max_num_joints,), value, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[-1] > 9:
                    values = arr[:, 9].reshape(-1)
                    if values.size == 1:
                        return np.full((self.max_num_joints,), float(values[0]), dtype=np.float32)
                    if values.size != self.max_num_joints:
                        raise ValueError("Tokenizer per-slot angle stats do not match max_num_joints")
                    return values.astype(np.float32)
                raise ValueError("Tokenizer angle stats must be shared [D] or per-slot [J,D]")
            if method == "zscore":
                token_mean = angle_slots("y_mean", 0.0)
                token_std = np.maximum(angle_slots("y_std", 1.0), 1e-6)
            else:
                lo_key, hi_key = ("y_min", "y_max") if method == "minmax" else ("y_q01", "y_q99")
                token_lo = angle_slots(lo_key, -1.0)
                token_hi = angle_slots(hi_key, 1.0)
                if np.any(token_hi <= token_lo):
                    raise ValueError("Tokenizer angle stats require hi > lo")
            self.tokenizer_stats_path = str(Path(self.tokenizer_stats_path).resolve())
        if token_lo is not None:
            token_lo = np.tile(token_lo, self.num_arms)
            token_hi = np.tile(token_hi, self.num_arms)
        if token_mean is not None:
            token_mean = np.tile(token_mean, self.num_arms)
            token_std = np.tile(token_std, self.num_arms)
        if self.normalization in {"q99", "minmax"}:
            if stats is None and stats_path is None:
                raise ValueError(f"{self.normalization} requires stats_path or stats")
            raw = stats
            if raw is None:
                with open(stats_path, encoding="utf-8") as handle:
                    raw = json.load(handle)
            if not isinstance(raw, Mapping):
                raise TypeError("continuous stats must be a mapping")
            raw = raw.get("action", raw)
            low_key, high_key = ("q01", "q99") if self.normalization == "q99" else ("min", "max")
            if not isinstance(raw, Mapping) or low_key not in raw or high_key not in raw:
                raise ValueError(f"continuous stats must contain action.{low_key} and action.{high_key}")
            q01 = np.asarray(raw[low_key], dtype=np.float32)
            q99 = np.asarray(raw[high_key], dtype=np.float32)
            if q01.shape != (self.action_dim,) or q99.shape != (self.action_dim,):
                raise ValueError(f"q01/q99 must have shape {(self.action_dim,)}")
            if (not np.isfinite(q01).all() or not np.isfinite(q99).all() or
                    np.any(q99 <= q01)):
                raise ValueError(f"continuous {low_key}/{high_key} must be finite with high>low for every slot")
            self.register_buffer("_q01", torch.as_tensor(q01), persistent=False)
            self.register_buffer("_q99", torch.as_tensor(q99), persistent=False)
        else:
            self.register_buffer("_q01", torch.empty(0), persistent=False)
            self.register_buffer("_q99", torch.empty(0), persistent=False)
        if token_lo is not None:
            self.register_buffer("_token_angle_lo", torch.as_tensor(token_lo), persistent=False)
            self.register_buffer("_token_angle_hi", torch.as_tensor(token_hi), persistent=False)
            self.register_buffer("_token_angle_mean", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_std", torch.empty(0), persistent=False)
        elif token_mean is not None:
            self.register_buffer("_token_angle_lo", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_hi", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_mean", torch.as_tensor(token_mean), persistent=False)
            self.register_buffer("_token_angle_std", torch.as_tensor(token_std), persistent=False)
        else:
            self.register_buffer("_token_angle_lo", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_hi", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_mean", torch.empty(0), persistent=False)
            self.register_buffer("_token_angle_std", torch.empty(0), persistent=False)
        payload = {"kind": "continuous", "num_arms": self.num_arms,
                   "max_num_joints": self.max_num_joints,
                   "angle_low": self.angle_low, "angle_high": self.angle_high,
                   "value_low": self.value_low.tolist(), "value_high": self.value_high.tolist(),
                   "action_representation": self.action_representation,
                   "normalization": self.normalization,
                   "q01": None if q01 is None or self.normalization != "q99" else q01.tolist(),
                   "q99": None if q99 is None or self.normalization != "q99" else q99.tolist(),
                   "min": None if q01 is None or self.normalization != "minmax" else q01.tolist(),
                   "max": None if q99 is None or self.normalization != "minmax" else q99.tolist(),
                   "stats_path": self.stats_path,
                   "layout": "arm_major_joints_then_gripper"}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = ActionTokenSpec(
            kind="continuous", num_arms=self.num_arms,
            max_num_joints=self.max_num_joints,
            arm_token_count=self.max_num_joints + 1,
            vocab_sizes=tuple(1 for _ in range(self.action_dim)),
            layout="arm_major_joints_then_gripper", fingerprint=fingerprint)

    def _joint_token_mask(self, m):
        return m

    def encode_actions(self, examples):
        raise RuntimeError("ContinuousActionCodec has no action-token encoding")

    def decode_actions(self, ids, examples, valid_mask=None):
        raise RuntimeError("ContinuousActionCodec has no action-token decoding")

    @property
    def tokens_per_frame(self):
        return self.action_dim

    def normalization_metadata(self):
        ranged = self.normalization in {"q99", "minmax"}
        low_name, high_name = ("q01", "q99") if self.normalization == "q99" else ("min", "max")
        result = {"mode": self.normalization, "stats_path": self.stats_path,
                  "tokenizer_stats_path": self.tokenizer_stats_path,
                  "tokenizer_normalization_method": self._tokenizer_norm_method,
                  low_name: self._q01.detach().cpu().tolist() if ranged else None,
                  high_name: self._q99.detach().cpu().tolist() if ranged else None}
        return result

    def continuous_targets(self, examples):
        """Return normalized targets and an exact [B,H,D] physical mask."""
        u, m, q, g, tv = self._action_inputs(examples)
        valid_q = tv[:, :, None, None] & m[:, None]
        valid_g = tv[:, :, None] & m.any(-1)[:, None]
        lo = torch.as_tensor(self.value_low, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        hi = torch.as_tensor(self.value_high, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
        # q99-normalized joint data can exceed nominal joint limits by a small
        # simulator/solver tolerance; keep raw values for normalization.
        tolerance = 1e-3 if self.normalization == "q99" and self.action_representation != "ee" else 1e-6
        if self.normalization != "q99" and bool(((q < lo - tolerance) | (q > hi + tolerance))[valid_q].any()):
            raise ValueError("Continuous valid action value is outside configured bounds")
        if bool(((g < 0) | (g > 1))[valid_g].any()):
            raise ValueError("Continuous valid gripper is outside [0,1]")
        physical = torch.cat([q, g[..., None]], dim=-1).flatten(2)
        if self.normalization in {"q99", "minmax"}:
            lo = self._q01.to(device=physical.device, dtype=physical.dtype)
            hi = self._q99.to(device=physical.device, dtype=physical.dtype)
            values = 2 * (physical - lo) / (hi - lo) - 1
            # q99 keeps rare extremes inside the flow objective. minmax already
            # spans every recorded value, so it is not clipped again.
            if self.normalization == "q99":
                values = values.clamp(-2.2, 2.2)
        elif self.normalization == "tokenizer_stats":
            qflat = q.flatten(2)
            if self._tokenizer_norm_method == "zscore":
                qn = (qflat - self._token_angle_mean.to(q.device, q.dtype)) / self._token_angle_std.to(q.device, q.dtype)
            else:
                lo = self._token_angle_lo.to(q.device, q.dtype)
                hi = self._token_angle_hi.to(q.device, q.dtype)
                qn = 2 * (qflat - lo) / (hi - lo) - 1
            qn = qn.reshape_as(q)
            gn = 2 * g - 1
            values = torch.cat([qn, gn[..., None]], dim=-1).flatten(2)
        else:
            lo = torch.as_tensor(self.value_low, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
            hi = torch.as_tensor(self.value_high, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
            qn = 2 * (q - lo) / (hi - lo) - 1
            gn = 2 * g - 1
            values = torch.cat([qn, gn[..., None]], dim=-1).flatten(2)
        mask = torch.cat([valid_q, valid_g[..., None]], dim=-1).flatten(2)
        values = values.masked_fill(~mask, 0)
        return values, mask

    def decode_continuous(self, normalized, examples, valid_mask):
        if normalized.ndim != 3 or normalized.shape[-1] != self.action_dim:
            raise ValueError("Continuous output has the wrong shape")
        valid_mask = binary_bool_mask(valid_mask, device=normalized.device,
                                      name="continuous_action_valid_mask")
        if valid_mask.shape != normalized.shape:
            raise ValueError("Continuous output mask has the wrong shape")
        _, m = self.geometry(examples)
        allowed = self.valid_tokens(examples, normalized.shape[1])
        if valid_mask.shape != allowed.shape or bool((valid_mask & ~allowed).any()):
            raise ValueError("Continuous output cannot activate padded joints or arms")
        if not bool(torch.isfinite(normalized[valid_mask]).all()):
            raise ValueError("Nonfinite valid continuous output")
        values = normalized.masked_fill(~valid_mask, 0)
        # Match continuous_targets and the dataset exactly:
        # [arm0 joints, arm0 gripper, arm1 joints, arm1 gripper].
        if self.normalization in {"q99", "minmax"}:
            lo = self._q01.to(device=values.device, dtype=values.dtype)
            hi = self._q99.to(device=values.device, dtype=values.dtype)
            values = .5 * (values + 1) * (hi - lo) + lo
        shaped = values.reshape(*values.shape[:2], self.num_arms, self.max_num_joints + 1)
        q, g = shaped[..., :-1], shaped[..., -1]
        if self.normalization == "tokenizer_stats":
            qflat = q.flatten(2)
            if self._tokenizer_norm_method == "zscore":
                qflat = (qflat * self._token_angle_std.to(values.device, values.dtype)
                         + self._token_angle_mean.to(values.device, values.dtype))
            else:
                lo = self._token_angle_lo.to(values.device, values.dtype)
                hi = self._token_angle_hi.to(values.device, values.dtype)
                qflat = .5 * (qflat + 1) * (hi - lo) + lo
            q = qflat.reshape_as(q)
            g = (g + 1) * .5
        shaped_valid = valid_mask.reshape_as(shaped)
        if self.normalization == "tokenizer_stats":
            lo = torch.as_tensor(self.value_low, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
            hi = torch.as_tensor(self.value_high, device=q.device, dtype=q.dtype).reshape(1, 1, 1, -1)
            q = q.clamp_min(lo).clamp_max(hi)
        q = q.masked_fill(~shaped_valid[..., :-1], 0)
        g = g.clamp(0, 1).masked_fill(~shaped_valid[..., -1], 0)
        return self._result(q, g, m, valid_mask)
