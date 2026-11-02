"""Observation-only ice estimation, with straight and curved learning gates.

Inference accepts observations only, never a world, a simulator snapshot, or
true ice parameters. Board forecasts include collisions; persistent game-time
updates and UI integration are later gates. Earlier baselines are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from physics import RADIUS


@dataclass(frozen=True)
class GlideObservation:
    time: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    angular_velocity: float = 0.
    held: bool = False
    contact: bool = False
    sweeping: bool = False
    # Executed sweep duration since release, not a future sweep command.
    sweep_seconds: float = 0.


def observed_deceleration(window: tuple[GlideObservation, ...]) -> float:
    """Fit a speed slope; no gravity or friction labels enter this feature."""
    if len(window) < 6:
        raise ValueError("At least six free-glide observations are required.")
    if any(o.held or o.contact or o.sweeping for o in window):
        raise ValueError("This first model excludes held, contact, and swept windows.")
    times = np.array([o.time for o in window])
    velocities = np.array([o.velocity for o in window])
    positions = np.array([o.position for o in window])
    omega = np.array([o.angular_velocity for o in window])
    if (velocities.shape != (len(window), 2) or positions.shape != velocities.shape
            or not all(np.isfinite(a).all() for a in (times, velocities, positions, omega))):
        raise ValueError("Observations must contain finite planar kinematics.")
    if np.any(np.diff(times) <= 0) or times[-1] - times[0] < .2:
        raise ValueError("Use increasing timestamps spanning at least 0.2 seconds.")
    if np.any(np.abs(omega) > .05) or np.any(velocities[:, 0] <= .15) or np.any(np.abs(velocities[:, 1]) > .1):
        raise ValueError("This first model requires forward, straight, non-stopping motion.")
    speed = np.linalg.norm(velocities, axis=1)
    centered = times - times.mean()
    return float(-np.dot(centered, speed - speed.mean()) / np.dot(centered, centered))


@dataclass(frozen=True)
class FrictionRegressor:
    feature_mean: float
    feature_scale: float
    intercept: float
    coefficient: float
    regularization: float
    friction_bounds: tuple[float, float] = (.02, .09)

    @classmethod
    def fit(cls, features, labels, regularization=1e-6):
        x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if (x.ndim != 1 or x.shape != y.shape or len(x) < 3
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(regularization) or regularization < 0
                or np.any((y < .02) | (y > .09))):
            raise ValueError("Invalid training data or ridge penalty.")
        mean, scale = float(x.mean()), float(x.std())
        if scale < 1e-10:
            raise ValueError("Training must contain different ice conditions.")
        design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
        penalty = np.diag([0., regularization])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(mean, scale, float(coefficients[0]), float(coefficients[1]), float(regularization))

    def estimate(self, window: tuple[GlideObservation, ...]) -> float:
        feature = observed_deceleration(window)
        value = self.intercept + self.coefficient * (feature - self.feature_mean) / self.feature_scale
        return float(np.clip(value, *self.friction_bounds))

    def save(self, path: Path):
        payload = {"schema_version": 1, "scope": "straight_free_glide_friction_only",
                   "feature": "negative_OLS_speed_slope_m_s2", **self.__dict__}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text())
        if (payload.pop("schema_version") != 1
                or payload.pop("scope") != "straight_free_glide_friction_only"
                or payload.pop("feature") != "negative_OLS_speed_slope_m_s2"):
            raise ValueError("Unsupported friction model format.")
        payload["friction_bounds"] = tuple(payload["friction_bounds"])
        model = cls(**payload)
        if (not all(np.isfinite(v) for v in (model.feature_mean, model.feature_scale, model.intercept,
                                            model.coefficient, model.regularization, *model.friction_bounds))
                or model.feature_scale <= 0 or model.regularization < 0
                or model.friction_bounds != (.02, .09)):
            raise ValueError("Invalid saved friction model.")
        return model


def predict_straight_endpoint(observation: GlideObservation, friction: float) -> dict:
    """Analytic Coulomb rollout from measured state, not a MuJoCo clone.

    The first experiment intentionally assumes the same constant-drag law as
    the simulator. Outside predictions report boundary crossing, not a stop.
    This function is not a collision or curve forecast.
    """
    if not np.isfinite(friction) or not .02 <= friction <= .09:
        raise ValueError("Invalid friction estimate.")
    if observation.held or observation.contact or observation.sweeping or abs(observation.angular_velocity) > .05:
        raise ValueError("This forecast requires straight free glide without sweeping.")
    position, velocity = np.asarray(observation.position), np.asarray(observation.velocity)
    if (position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all()
            or not np.isfinite(velocity).all() or velocity[0] <= 0 or abs(velocity[1]) > .1):
        raise ValueError("Invalid forward-glide state.")
    if np.any(position < [RADIUS, -.75 + RADIUS]) or np.any(position > [4 - RADIUS, .75 - RADIUS]):
        raise ValueError("Observation must be inside the rink.")
    displacement = velocity * float(np.linalg.norm(velocity)) / (2 * friction * 9.81)
    fraction = 1.
    for axis, (low, high) in enumerate(((RADIUS, 4 - RADIUS), (-.75 + RADIUS, .75 - RADIUS))):
        if displacement[axis] > 0:
            fraction = min(fraction, float((high - position[axis]) / displacement[axis]))
        elif displacement[axis] < 0:
            fraction = min(fraction, float((low - position[axis]) / displacement[axis]))
    return {"position": (position + fraction * displacement).tolist(), "alive": fraction >= 1.,
            "unbounded_stop": (position + displacement).tolist(), "assumed_friction": float(friction)}


@dataclass(frozen=True)
class IceFeatures:
    deceleration: float
    curvature: float | None
    curvature_standard_error: float | None


def _slope(x, y):
    xc = x - x.mean()
    denominator = float(xc @ xc)
    if denominator < 1e-12:
        raise ValueError("Insufficient variation in observation window.")
    slope = float(xc @ (y - y.mean()) / denominator)
    residual = y - y.mean() - slope * xc
    standard_error = float(np.sqrt((residual @ residual) / (len(x) - 2) / denominator))
    return slope, standard_error


def ice_features(window: tuple[GlideObservation, ...]) -> IceFeatures:
    """Normalize observed motion by known executed sweep and spin effects.

    Sweep strengths (20% drag / 30% curl), tanh spin response, and speed scale
    are known game laws. Ice coefficients themselves are unknown at inference.
    """
    if len(window) < 6 or any(o.held or o.contact for o in window):
        raise ValueError("Use at least six observations without grasp or contact.")
    times = np.array([o.time for o in window])
    velocity = np.array([o.velocity for o in window])
    position = np.array([o.position for o in window])
    omega = np.array([o.angular_velocity for o in window])
    sweep = np.array([o.sweep_seconds for o in window])
    if (velocity.shape != (len(window), 2) or position.shape != velocity.shape
            or not all(np.isfinite(a).all() for a in (times, velocity, position, omega, sweep))):
        raise ValueError("Use finite planar kinematics and sweep history.")
    dt, ds = np.diff(times), np.diff(sweep)
    if (np.any(dt <= 0) or times[-1] - times[0] < .2 or np.any(sweep < 0)
            or np.any(ds < -1e-9) or np.any(ds > dt + 1e-9)):
        raise ValueError("Invalid timestamps or cumulative executed sweep durations.")
    if any(o.sweeping and delta <= 0 for o, delta in zip(window[1:], ds)):
        raise ValueError("Swept samples require the executed sweep history.")
    speed = np.linalg.norm(velocity, axis=1)
    if np.any(speed < .15):
        raise ValueError("Near-stop windows cannot reliably identify ice.")
    drag_clock = times - times[0] - .20 * (sweep - sweep[0])
    deceleration = -_slope(drag_clock, speed)[0]
    spin_factor = np.tanh(omega) / (speed + .1)
    # Integrate heading exposure using observed kinematics and known controls.
    exposure = np.r_[0., np.cumsum(.5 * (spin_factor[:-1] + spin_factor[1:]) * (dt - .30 * ds))]
    if float(np.mean(np.abs(np.tanh(omega)))) < .15 or abs(exposure[-1]) < .025:
        return IceFeatures(deceleration, None, None)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    curvature, error = _slope(exposure, heading)
    return IceFeatures(deceleration, curvature, error)


def _fit_calibration(features, targets, penalty):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(penalty) or penalty < 0 or x.std() < 1e-10):
        raise ValueError("Calibration requires varied, finite training data.")
    mean, scale = float(x.mean()), float(x.std())
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights = np.linalg.solve(design.T @ design + np.diag([0., penalty]), design.T @ y)
    return mean, scale, float(weights[0]), float(weights[1])


def _calibrated(feature, calibration):
    mean, scale, intercept, coefficient = calibration
    return intercept + coefficient * (feature - mean) / scale


@dataclass(frozen=True)
class IceEstimate:
    friction: float
    curl: float
    curl_observable: bool = False
    curl_weight: float = 0.
    curl_standard_error: float | None = None


@dataclass(frozen=True)
class IceRegressor:
    friction_calibration: tuple[float, float, float, float]
    curl_calibration: tuple[float, float, float, float]
    prior_friction: float
    prior_curl: float
    curl_prior_variance: float
    regularization: float

    @classmethod
    def fit(cls, windows, friction_labels, curl_labels, regularization=1e-6):
        if not (len(windows) == len(friction_labels) == len(curl_labels)):
            raise ValueError("Mismatched training windows and labels.")
        mu, kappa = np.asarray(friction_labels), np.asarray(curl_labels)
        if (not np.isfinite(mu).all() or not np.isfinite(kappa).all()
                or np.any((mu < .02) | (mu > .09)) or np.any((kappa < 0) | (kappa > .1))):
            raise ValueError("Invalid ice training labels.")
        features = [ice_features(w) for w in windows]
        observed = [i for i, f in enumerate(features) if f.curvature is not None]
        friction_fit = _fit_calibration([f.deceleration for f in features], mu, regularization)
        curl_fit = _fit_calibration([features[i].curvature for i in observed], kappa[observed], regularization)
        return cls(friction_fit, curl_fit, float(mu.mean()), float(kappa.mean()),
                   float(np.var(kappa)), float(regularization))

    def estimate(self, window, previous: IceEstimate | None = None) -> IceEstimate:
        feature = ice_features(window)
        friction = float(np.clip(_calibrated(feature.deceleration, self.friction_calibration), .02, .09))
        prior = self.prior_curl if previous is None else previous.curl
        if not np.isfinite(prior) or not 0 <= prior <= .1:
            raise ValueError("Invalid previous curl estimate.")
        if feature.curvature is None:
            return IceEstimate(friction, prior)
        raw = _calibrated(feature.curvature, self.curl_calibration)
        standard_error = abs(self.curl_calibration[3] / self.curl_calibration[1]) * feature.curvature_standard_error
        # Shrink a weak/noisy turn signal toward the existing estimate. This
        # is a heuristic residual-based weight, not a calibrated confidence CI.
        variance = max(self.curl_prior_variance, 1e-12)
        weight = float(variance / (variance + standard_error ** 2))
        curl = float(np.clip(prior + weight * (raw - prior), 0., .1))
        return IceEstimate(friction, curl, True, weight, float(standard_error))

    def save(self, path):
        payload = {"schema_version": 1, "scope": "curved_free_glide_ice_v1", **self.__dict__}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.pop("schema_version") != 1 or payload.pop("scope") != "curved_free_glide_ice_v1":
            raise ValueError("Unsupported ice model format.")
        for name in ("friction_calibration", "curl_calibration"):
            values = tuple(payload[name])
            if len(values) != 4 or not np.isfinite(values).all() or values[1] <= 0:
                raise ValueError("Invalid saved calibration.")
            payload[name] = values
        model = cls(**payload)
        if (not all(np.isfinite(v) for v in (model.prior_friction, model.prior_curl, model.curl_prior_variance, model.regularization))
                or not .02 <= model.prior_friction <= .09 or not 0 <= model.prior_curl <= .1
                or model.curl_prior_variance < 0 or model.regularization < 0):
            raise ValueError("Invalid saved ice model.")
        return model


def predict_free_glide(observation: GlideObservation, estimate: IceEstimate) -> dict:
    """Single stone in a new MuJoCo world, assuming no sweep from now on.

    Only measured position, velocity and spin are restored. No source world's
    model, forces, accelerations, warm-start, truth, or future commands enter.
    Velocity initialization here is observation restoration in the forecast;
    the actual shot still gets its momentum from robot actuators.
    """
    import mujoco
    from physics import CurlingWorld, ShotCommand

    position, velocity = np.asarray(observation.position), np.asarray(observation.velocity)
    if (observation.held or observation.contact or position.shape != (2,) or velocity.shape != (2,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not np.isfinite(observation.angular_velocity)
            or np.any(position < [RADIUS, -.75 + RADIUS]) or np.any(position > [4 - RADIUS, .75 - RADIUS])):
        raise ValueError("Forecast requires an in-rink, free-gliding observation.")
    world = CurlingWorld(friction=estimate.friction, curl=estimate.curl)
    world._place(0, position)
    address = world.qvel_addresses[0]
    world.data.qvel[address:address + 3] = [*velocity, observation.angular_velocity]
    # The robot does not contact a released stone; keep it at home. Absolute
    # yaw is immaterial for the circular collision geometry and planar inertia.
    world.command, world.current = ShotCommand(), 0
    world.phase, world.start_time, world.release = "SLIDING", 0., 0.
    world.data.time = 2.
    mujoco.mj_forward(world.model, world.data)
    result = world.run_until_rest()["stones"][0]
    return {"position": result["position"][:2], "alive": result["alive"],
            "assumption": "no sweep from the observed state onward"}


def predict_board(observation, estimate: IceEstimate, command=None, include_trace=False):
    """Forecast all played stones with estimated ice and no future sweeping.

    READY requires a proposed next shot. During a shot, only its already
    executed command may be used. The source world is not an argument.
    """
    from physics import CurlingWorld

    if observation.phase == "READY" and command is None:
        raise ValueError("A pre-shot forecast requires a proposed shot command.")
    if observation.phase != "READY" and command is not None and command != observation.command:
        raise ValueError("Cannot replace a command after the shot has started.")
    world = CurlingWorld(estimate.friction, estimate.curl)
    world.restore_observation(observation)
    if world.phase == "READY":
        world.start_shot(command)
    trace = []
    callback = (lambda w: trace.append(w.positions().tolist())) if include_trace else None
    result = world.run_until_rest(callback=callback)
    result["assumption"] = "no sweep from the observed state onward"
    result["observed_time"] = observation.time
    return result, trace


class OnlineIceEstimator:
    """Update ice estimates from a rolling free-glide window, never truth.

    Model weights stay fixed. A new shot clears its observation window while
    retaining estimates; retry resets both. Contact-contaminated intervals and
    near-stop motion are excluded, including contacts between sample instants.
    """
    def __init__(self, model: IceRegressor, enabled=True, alpha=.35):
        if not 0 < alpha <= 1:
            raise ValueError("Update weight must be in (0, 1].")
        self.model, self.enabled, self.alpha = model, enabled, alpha
        self.reset()

    def reset(self):
        self.estimate = IceEstimate(self.model.prior_friction, self.model.prior_curl)
        self.updates = 0
        self.events = []
        self.start_shot()

    def start_shot(self):
        self.window = []
        self.last_update = -float("inf")

    def observe(self, observation: GlideObservation):
        if (observation.held or observation.contact or np.linalg.norm(observation.velocity) < .15):
            self.window.clear()
            return False
        if self.window and observation.time - self.window[-1].time > .061:
            self.window.clear()
        self.window.append(observation)
        self.window = self.window[-16:]
        if (not self.enabled or len(self.window) < 16 or self.window[-1].time - self.window[0].time < .59
                or observation.time - self.last_update < .199):
            return False
        candidate = self.model.estimate(tuple(self.window), self.estimate)
        old = self.estimate
        self.estimate = IceEstimate(
            (1 - self.alpha) * old.friction + self.alpha * candidate.friction,
            (1 - self.alpha) * old.curl + self.alpha * candidate.curl,
            candidate.curl_observable, candidate.curl_weight, candidate.curl_standard_error)
        self.updates += 1
        self.last_update = observation.time
        self.events.append({"time": observation.time, "friction": self.estimate.friction,
                            "curl": self.estimate.curl, "curl_observable": self.estimate.curl_observable})
        return True
