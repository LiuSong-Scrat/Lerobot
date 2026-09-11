"""Opt-in solver-only experiments; never patch the installed MuJoCo engine."""

PROFILES = ("default", "noslip_v1")
_STATE_KEY = "_song_contact_solver_override"
_OPTION_FIELDS = (
    "timestep", "integrator", "solver", "cone", "impratio", "iterations",
    "tolerance", "noslip_iterations", "noslip_tolerance", "enableflags", "disableflags",
)


def validate_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown contact solver profile: {profile!r}; expected {PROFILES}")
    return profile


def _snapshot(opt):
    return {name: getattr(opt, name).item() if hasattr(getattr(opt, name), "item")
            else getattr(opt, name) for name in _OPTION_FIELDS}


def restore_contact_solver(env):
    """Undo only our change, and only on the model instance we changed."""
    state = vars(env).pop(_STATE_KEY, None)
    if state is not None:
        model, previous = state
        if env.sim.model._model is model:
            model.opt.noslip_iterations = previous


def apply_contact_solver(env, profile="default"):
    """Call AFTER init/dummy steps; no state writes, stepping, or new observations."""
    validate_profile(profile)
    restore_contact_solver(env)
    model = env.sim.model._model
    before = _snapshot(model.opt)
    if profile == "noslip_v1":
        import mujoco

        if mujoco.mj_versionString() != "3.3.4":
            raise ValueError("noslip_v1 is scoped to MuJoCo 3.3.4; do not silently change engines")
        setattr(env, _STATE_KEY, (model, int(model.opt.noslip_iterations)))
        model.opt.noslip_iterations = 3
    return {
        "profile": profile,
        "applied_after_initialization": profile != "default",
        "scope": "all_contacts_during_rollout" if profile != "default" else "unchanged",
        "before": before,
        "after": _snapshot(model.opt),
        "changed_fields": [name for name in before if before[name] != getattr(model.opt, name)],
    }
