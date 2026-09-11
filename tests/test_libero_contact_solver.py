"""Solver profile contract tests; no policy or task replay."""
import importlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from benchmarks.song_real_libero.scripts.libero_setting import libero_contact_solver as cs


def environment():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="slide"/>'
        '<geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>'
    )
    return SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=model)))


def test_default_does_not_modify_model():
    env = environment()
    record = cs.apply_contact_solver(env)
    assert record['changed_fields'] == []
    assert record['before'] == record['after']
    json.dumps(record)


def test_only_noslip_changes_and_restore_is_idempotent(monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_versionString', lambda: '3.3.4')
    env = environment()
    m = env.sim.model._model
    arrays = {name: getattr(m, name).copy() for name in
              ['geom_friction', 'geom_solref', 'geom_solimp', 'geom_size',
               'actuator_forcerange', 'dof_damping', 'dof_armature']}
    record = cs.apply_contact_solver(env, 'noslip_v1')
    assert record['changed_fields'] == ['noslip_iterations']
    assert m.opt.noslip_iterations == 3
    assert record['applied_after_initialization']
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(m, name), value)
    cs.apply_contact_solver(env, 'noslip_v1')
    cs.restore_contact_solver(env)
    cs.restore_contact_solver(env)
    assert m.opt.noslip_iterations == record['before']['noslip_iterations']


def test_reset_model_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_versionString', lambda: '3.3.4')
    env = environment()
    cs.apply_contact_solver(env, 'noslip_v1')
    env.sim.model._model = environment().sim.model._model
    env.sim.model._model.opt.noslip_iterations = 7
    cs.restore_contact_solver(env)
    assert env.sim.model._model.opt.noslip_iterations == 7


def test_wrong_version_fails_without_changes(monkeypatch):
    monkeypatch.setattr(mujoco, 'mj_versionString', lambda: '3.12.0')
    env = environment()
    with pytest.raises(ValueError, match='3.3.4'):
        cs.apply_contact_solver(env, 'noslip_v1')
    assert env.sim.model._model.opt.noslip_iterations == 0


def test_invalid_profile():
    with pytest.raises(ValueError):
        cs.apply_contact_solver(environment(), 'typo')


@pytest.mark.parametrize('base_config', [{}, {'control': {'control_freq': 5}}, {'world_to_ego_causal_ablation': True}])
def test_protocol_and_episode_metadata(monkeypatch, base_config):
    monkeypatch.setenv('SONG_LIBERO_ENV_WORKER', '1')
    ev = importlib.import_module('benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_eval')
    base = ev.evaluation_protocol_for_config(base_config)
    cfg = {**base_config, 'contact_solver_profile': 'noslip_v1'}
    result = ev.evaluation_protocol_for_config(cfg)
    assert not result['benchmark_comparable']
    assert result['base_protocol'] == base['name']
    assert result['contact_solver_profile'] == 'noslip_v1'
    assert ev.evaluation_protocol_for_config(base_config) == base
    compact = ev.compact_episode_record({'contact_solver_settings': {'profile': 'noslip_v1'}}, 0, None)
    assert compact['contact_solver_settings']['profile'] == 'noslip_v1'


def test_cli(monkeypatch):
    monkeypatch.setenv('SONG_LIBERO_ENV_WORKER', '1')
    ev = importlib.import_module('benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_eval')
    monkeypatch.setattr('sys.argv', ['eval', '--contact-solver-profile', 'noslip_v1'])
    assert ev.parse_args().contact_solver_profile == 'noslip_v1'
