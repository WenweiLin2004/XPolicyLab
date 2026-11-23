"""Source metadata inspection used only by the RoboTwin converter."""
from pathlib import Path
import json
import os
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from robomorpher.utils.data.process_data.bench_paths import stored_path
from robomorpher.utils.data.urdf_features import prepare_tokenizer_chain

class RoboTwinSourceIndex:
    """Conversion-only native HDF5 episode/URDF index. Not a training Dataset."""

    def __init__(self, data_root, *, max_num_joints=7, urdf_root=None, camera_names=('cam_high',), joint_names=None, embodiment_urdf_map=None, allow_missing_urdf=False, embodiment=None):
        self.data_root = Path(data_root).expanduser().resolve()
        self.max_num_joints = int(max_num_joints)
        self.urdf_root = Path(urdf_root).expanduser().resolve() if urdf_root else None
        self.camera_names = tuple(camera_names)
        self.joint_names = dict(joint_names or {})
        self.embodiment_urdf_map = dict(embodiment_urdf_map or {})
        self.allow_missing_urdf = bool(allow_missing_urdf)
        self._chain_cache = {}
        self._registered_bundles = {}
        self.robotwin_root = None
        if self.urdf_root and self.urdf_root.parent.name == "assets":
            root = self.urdf_root.parent.parent
            if (self.urdf_root.name == "embodiments"
                    and (root / "env_cfg/task_config/_embodiment_config.yml").is_file()):
                self.robotwin_root = root
        # XPolicyLab collections may link whole task or environment directories.
        # Visit each physical directory once so aliases and symlink cycles are safe.
        paths = []
        visited = set()
        for directory, subdirs, files in os.walk(self.data_root, followlinks=True):
            physical = Path(directory).resolve()
            if physical in visited:
                subdirs.clear()
                continue
            visited.add(physical)
            subdirs.sort()
            paths.extend(Path(directory) / name for name in sorted(files) if name.endswith(".hdf5"))
        if embodiment:
            key = self._robot_key(embodiment)
            paths = [path for path in paths if self._robot_key(path.parent.parent.name) == key]
        if not paths:
            raise FileNotFoundError(f'No native HDF5 episodes under {self.data_root}')
        self.episodes = [self._inspect_episode(path) for path in paths]
        if any((ep['source_length'] < 1 for ep in self.episodes)):
            raise ValueError('Cannot convert an empty episode')
    _CAMERA_ALIASES = {'cam_high': 'cam_head', 'cam_head': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist', 'cam_third_view': 'cam_third_view'}
    _ROBOT_ALIASES = {'aloha': 'aloha-agilex', 'aloha_agilex': 'aloha-agilex', 'arx': 'ARX-X5', 'arx_x5': 'ARX-X5', 'arx_x5_arx_x5': 'ARX-X5', 'x5': 'ARX-X5', 'franka': 'franka-panda', 'franka_panda': 'franka-panda', 'franka_panda_franka_panda': 'franka-panda', 'piper': 'piper', 'piper_piper': 'piper', 'ur5': 'ur5-wsg', 'ur5_wsg': 'ur5-wsg', 'ur5_wsg_ur5_wsg': 'ur5-wsg'}
    _DEFAULT_JOINT_NAMES = {'ARX-X5': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'], 'piper': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'], 'franka-panda': [f'panda_joint{i}' for i in range(1, 8)], 'ur5-wsg': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']}

    @staticmethod
    def _decode_text(value):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        return value.decode('utf-8') if isinstance(value, (bytes, np.bytes_)) else str(value)

    def _robot_key(self, raw):
        if self.robotwin_root and raw not in self.embodiment_urdf_map:
            from robot_metadata import resolve_embodiment

            spec = resolve_embodiment(self.robotwin_root, raw)
            if len(spec) == 1 or spec[0] == spec[1]:
                return spec[0]
            return f"{spec[0]}+{spec[1]}"
        name = str(raw).lower().replace('-', '_')
        if name in self._ROBOT_ALIASES:
            return self._ROBOT_ALIASES[name]
        for key, value in self._ROBOT_ALIASES.items():
            if key in name:
                return value
        return str(raw)

    def _find_urdf(self, raw_embodiment, key):
        explicit = self.embodiment_urdf_map.get(raw_embodiment, self.embodiment_urdf_map.get(key))
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        if self.urdf_root:
            root = self.urdf_root
            for name in (raw_embodiment, key):
                base = root / name
                if base.is_file():
                    candidates.append(base)
                elif base.is_dir():
                    candidates.extend(sorted(base.rglob('*.urdf')))
        filtered = [x.resolve() for x in candidates if x.is_file() and 'moveit' not in str(x).lower() and ('no_gripper' not in x.name.lower())]
        if filtered:
            return (filtered[0], key)
        if self.allow_missing_urdf:
            return (None, key)
        raise FileNotFoundError(f'No URDF for native embodiment {raw_embodiment!r} (normalized {key!r}). Set data.urdf_root or data.embodiment_urdf_map.')

    def _names_for(self, raw_embodiment, key, side, dim, registered_names):
        for name in (raw_embodiment, key, f'{key}:{side}', side):
            if name in self.joint_names:
                names = list(self.joint_names[name])
                if len(names) != dim:
                    raise ValueError(f'Joint-name count for {name!r} is {len(names)}, expected {dim}')
                return names
        if registered_names is not None:
            names = registered_names
        elif key == 'aloha-agilex':
            prefix = 'fl' if side == 'left' else 'fr'
            names = [f'{prefix}_joint{i}' for i in range(1, dim + 1)]
        else:
            names = self._DEFAULT_JOINT_NAMES.get(key)
        if names is None or len(names) != dim:
            raise ValueError(f'No ordered joint_names for native embodiment {raw_embodiment!r}, side={side}, dim={dim}; provide data.joint_names')
        return names

    def _make_chain(self, raw_embodiment, side, dim):
        cache_key = (str(raw_embodiment), str(side), int(dim))
        if cache_key in self._chain_cache:
            return self._chain_cache[cache_key]
        if dim > self.max_num_joints:
            raise ValueError(f"{raw_embodiment} {side} arm has {dim} joints; the dataset has {self.max_num_joints} slots")
        key = self._robot_key(raw_embodiment)
        registered_names = None
        if self.robotwin_root and not any(name in self.embodiment_urdf_map for name in (raw_embodiment, key)):
            from robot_metadata import resolve_robot_bundle

            if raw_embodiment not in self._registered_bundles:
                self._registered_bundles[raw_embodiment] = resolve_robot_bundle(self.robotwin_root, raw_embodiment)
            paths, joint_names = self._registered_bundles[raw_embodiment]
            side_idx = 0 if side == "left" else 1
            urdf_path, registered_names = paths[side_idx], joint_names[side_idx]
        else:
            urdf_path, key = self._find_urdf(raw_embodiment, key)
        if urdf_path is None:
            result = (None, key, [], None)
            self._chain_cache[cache_key] = result
            return result
        names = self._names_for(raw_embodiment, key, side, dim, registered_names)
        root = ET.parse(urdf_path).getroot()
        available = {j.get('name') for j in root.findall('joint') if j.get('type') in {'revolute', 'continuous'} and j.find('mimic') is None}
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f'Joint names missing from {urdf_path}: {missing}')
        result = (prepare_tokenizer_chain(urdf_path, names, embed_arm_base=False), key, names, urdf_path)
        self._chain_cache[cache_key] = result
        return result

    def _instruction_list(self, f):
        key = 'instructions' if 'instructions' in f else 'instruction'
        if key not in f:
            raise ValueError('Native episode has no instruction/instructions key')
        raw = self._decode_text(f[key][()])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [raw]
        if isinstance(parsed, str):
            parsed = [parsed]
        if not isinstance(parsed, list) or not parsed or (not all((str(x).strip() for x in parsed))):
            raise ValueError('Native instructions must be a nonempty string or list')
        return [str(x) for x in parsed]

    def _camera_key(self, camera):
        native = self._CAMERA_ALIASES.get(camera, camera)
        return f'vision/{native}/colors'

    def _inspect_episode(self, path):
        with h5py.File(path, 'r') as f:
            if 'state' not in f or 'vision' not in f:
                raise ValueError(f'Not a native RoboTwin trajectory HDF5: {path}')
            instructions = self._instruction_list(f)
            (dimensions, state_keys) = ([], [])
            for side in ('left', 'right'):
                arm_key = f'{side}_arm_joint_states'
                grip_key = f'{side}_ee_joint_states'
                if f'state/{arm_key}' not in f:
                    (arm_key, grip_key) = ('arm_joint_states', 'ee_joint_states')
                if f'state/{arm_key}' not in f or f'state/{grip_key}' not in f:
                    raise ValueError(f'Missing native {side} arm/gripper state keys in {path}')
                arm = f[f'state/{arm_key}']
                grip = f[f'state/{grip_key}']
                if arm.ndim != 2 or grip.ndim != 2 or grip.shape[1] != 1 or (arm.shape[0] != grip.shape[0]):
                    raise ValueError(f'Invalid native state layout in {path} ({side})')
                dimensions.append(int(arm.shape[1]))
                state_keys.append((arm_key, grip_key))
            source_length = int(f[f'state/{state_keys[0][0]}'].shape[0])
            for (arm_key, grip_key) in state_keys:
                if f[f'state/{arm_key}'].shape[0] != source_length:
                    raise ValueError(f'Native arm state time mismatch in {path}')
                if f[f'state/{grip_key}'].shape[0] != source_length:
                    raise ValueError(f'Native gripper state time mismatch in {path}')
            for camera in self.camera_names:
                key = self._camera_key(camera)
                if key not in f or f[key].shape[0] != source_length:
                    raise ValueError(f'Missing native camera/time alignment: {key} in {path}')
            raw_embodiment = path.parent.parent.name
        features = np.zeros((2, self.max_num_joints, 6), dtype=np.float32)
        mask = np.zeros((2, self.max_num_joints), dtype=bool)
        (urdf_paths, names_by_side, key) = ([None, None], [[], []], None)
        if self.allow_missing_urdf:
            return dict(path=path, length=source_length, source_length=source_length, dimensions=dimensions, state_keys=state_keys, instructions=instructions, urdf_features=features, joint_valid_mask=mask, urdf_paths=urdf_paths, joint_names=names_by_side, embodiment=raw_embodiment, robot_key=key, schema='robotwin_native')
        for (arm, side) in enumerate(('left', 'right')):
            (chain, key, names, urdf_path) = self._make_chain(raw_embodiment, side, dimensions[arm])
            if chain is not None:
                features[arm, :dimensions[arm]] = chain['features']
                mask[arm, :dimensions[arm]] = True
                urdf_paths[arm] = stored_path(urdf_path)
            elif not self.allow_missing_urdf:
                raise RuntimeError('Unreachable missing-URDF branch')
            names_by_side[arm] = names
        length = source_length
        return dict(path=path, length=length, source_length=source_length, dimensions=dimensions, state_keys=state_keys, instructions=instructions, urdf_features=features, joint_valid_mask=mask, urdf_paths=urdf_paths, joint_names=names_by_side, embodiment=raw_embodiment, robot_key=key, schema='robotwin_native')
