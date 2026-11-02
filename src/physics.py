"""Small MuJoCo world shared by the curling PoC and its reference forecasts.

Only start-of-shot placement and out-of-play removal assign stone state.
The robot's actuators and a releasable grasp generate the delivery velocity.
Default stones move in the plane with explicit Coulomb drag and circular
contact. A full 3D diagnostic variant remains available as planar=False.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DT = .002
RADIUS = .07
HOUSE = np.array([3., 0.])
HOUSE_RADIUS = .45
STATE_SIGNATURE = mujoco.mjtState.mjSTATE_INTEGRATION
COLORS = ((1., .48, .08, 1.), (.1, .66, .78, 1.), (.65, .38, .86, 1.))


@dataclass(frozen=True)
class ShotCommand:
    angle: float = 0.
    power: float = 65.
    spin: int = 0

    def __post_init__(self):
        if not math.isfinite(self.angle) or not -15 <= self.angle <= 15:
            raise ValueError("Direction must be in [-15, 15] degrees.")
        if not math.isfinite(self.power) or not 1 <= self.power <= 100:
            raise ValueError("Power must be in [1, 100].")
        if self.spin not in (-1, 0, 1):
            raise ValueError("Spin must be -1, 0, or 1.")

    @property
    def speed(self):
        return .4 + 1.5 * (self.power - 1.) / 99.

    @property
    def omega(self):
        return self.spin * 1.5


@dataclass(frozen=True)
class StoneObservation:
    id: int
    alive: bool
    position: tuple[float, float]
    velocity: tuple[float, float]
    yaw: float
    angular_velocity: float


@dataclass(frozen=True)
class BoardObservation:
    """Kinematics and known game/control events; no hidden physics state.

    Stones contains played stones only, including previously removed stones.
    Robot joints are observable encoders in this simulation. Force, friction,
    acceleration, solver warm-start and model objects are deliberately absent.
    """
    time: float
    phase: str
    stones: tuple[StoneObservation, ...]
    robot_position: tuple[float, ...]
    robot_velocity: tuple[float, ...]
    current: int | None
    command: ShotCommand | None
    start_time: float
    release_time: float | None
    settle_time: float


def score_stone(position, active=True):
    if not active:
        return 0
    d = float(np.linalg.norm(np.asarray(position)[:2] - HOUSE))
    if d > HOUSE_RADIUS + RADIUS + 1e-9:
        return 0
    if d >= HOUSE_RADIUS - RADIUS - 1e-9:
        return 1
    raw = 1 + 99 * (1 - d / (HOUSE_RADIUS - RADIUS))
    return max(1, min(100, math.floor(raw + .5)))


class CurlingWorld:
    """One private model/data pair. Ice values are explicit physical parameters.

    This class does not provide learned estimates. The PoC uses known-parameter
    forecasts to measure simulator consistency separately from AI performance.
    """

    def __init__(self, friction=.05, curl=.03, planar=True):
        if not .02 <= friction <= .09 or not 0 <= curl <= .1:
            raise ValueError("Ice parameters outside the supported PoC range.")
        self.friction, self.curl = float(friction), float(curl)
        self.planar = bool(planar)
        scene_path = Path(__file__).with_name("game_scene.xml")
        self.parks = [np.array([-1., 2. + .3 * i, .026]) for i in range(3)]
        if self.planar:
            tree = ET.parse(scene_path)
            floor = tree.find(".//geom[@name='floor']")
            floor.set("contype", "0")
            floor.set("conaffinity", "0")
            contact = tree.find("contact")
            for pair in list(contact):
                if pair.get("name", "").startswith("ice"):
                    contact.remove(pair)
            for i in range(3):
                body = tree.find(f".//body[@name='stone{i}']")
                body.remove(body.find("freejoint"))
                # At fixed z, sphere-sphere contact is exactly the contact of
                # two planar circles. The rendered stone remains a cylinder.
                disk = body.find(f"geom[@name='disk{i}']")
                disk.set("type", "sphere")
                disk.set("size", ".07")
                disk.set("rgba", "0 0 0 0")
                ET.SubElement(body, "geom", {"class": "decor", "type": "cylinder", "size": ".07 .025",
                                             "rgba": " ".join(map(str, COLORS[i]))})
                ET.SubElement(body, "inertial", pos="0 0 0", mass=".2", diaginertia=".000286666667 .000286666667 .00049")
                for suffix, kind, axis in [("x", "slide", "1 0 0"), ("y", "slide", "0 1 0"), ("yaw", "hinge", "0 0 1")]:
                    ET.SubElement(body, "joint", name=f"stone{i}_{suffix}", type=kind, axis=axis)
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.original_alpha = self.model.geom_rgba[:, 3].copy()
        self.body_ids = [self.model.body(f"stone{i}").id for i in range(3)]
        self.disk_ids = [self.model.geom(f"disk{i}").id for i in range(3)]
        self.grasps = [self.model.equality(f"grasp{i}").id for i in range(3)]
        self.ice_pairs = [] if self.planar else [self.model.pair(f"ice{i}").id for i in range(3)]
        suffix = "x" if self.planar else "free"
        self.qpos_addresses = [int(self.model.joint(f"stone{i}_{suffix}").qposadr[0]) for i in range(3)]
        self.qvel_addresses = [int(self.model.joint(f"stone{i}_{suffix}").dofadr[0]) for i in range(3)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.played = np.zeros(3, dtype=bool)
        self.alive = np.zeros(3, dtype=bool)
        self.removed_positions = {}
        self.phase = "READY"
        self.command = None
        self.current = None
        self.start_time = 0.
        self.settle_time = 0.
        self.release = None
        self.last_release = None
        self.collisions = set()
        self.max_height = .026
        self.max_speed = 0.
        self.max_passive_energy = 0.
        self.sweeping = False
        self.error = None
        self._sync_visibility()
        for pair in self.ice_pairs:
            self.model.pair_friction[pair, :2] = self.friction
        mujoco.mj_forward(self.model, self.data)

    def _sync_visibility(self):
        for i, body in enumerate(self.body_ids):
            start, count = self.model.body_geomadr[body], self.model.body_geomnum[body]
            for geom in range(start, start + count):
                self.model.geom_rgba[geom, 3] = self.original_alpha[geom] if self.alive[i] else 0.

    def _place(self, index, xy):
        q, v = self.qpos_addresses[index], self.qvel_addresses[index]
        if self.planar:
            self.data.qpos[q:q + 3] = [float(xy[0]) - self.parks[index][0], float(xy[1]) - self.parks[index][1], 0.]
            self.data.qvel[v:v + 3] = 0.
        else:
            self.data.qpos[q:q + 7] = [float(xy[0]), float(xy[1]), .026, 1., 0., 0., 0.]
            self.data.qvel[v:v + 6] = 0.
        self.played[index] = self.alive[index] = True
        self._sync_visibility()
        mujoco.mj_forward(self.model, self.data)

    def place_fixture(self, index, xy):
        """Place an initially resting stone for documented PoC collision cases."""
        if self.phase != "READY" or self.played[index]:
            raise ValueError("Fixture placement is only allowed before a shot.")
        self._place(index, xy)

    def start_shot(self, command):
        if self.phase != "READY":
            raise ValueError("Wait for the current shot to stop.")
        available = np.flatnonzero(~self.played)
        if not len(available):
            raise ValueError("All three stones have been delivered.")
        for i in np.flatnonzero(self.alive):
            if np.linalg.norm(self.positions()[i][:2] - [.30, 0.]) < 2 * RADIUS + .01:
                raise ValueError("The delivery start is blocked; retry this game.")
        self.current = int(available[0])
        self.command = command
        self._place(self.current, [.30, 0.])
        self.data.eq_active[:] = 0
        self.data.eq_active[self.grasps[self.current]] = 1
        self.start_time = float(self.data.time)
        self.settle_time = 0.
        self.release = None
        self.last_release = None
        self.collisions = set()
        self.max_height, self.max_speed = .026, 0.
        self.max_passive_energy = 0.
        self.phase = "DELIVERY"
        mujoco.mj_forward(self.model, self.data)

    def positions(self):
        return np.array([self.data.xpos[i].copy() for i in self.body_ids])

    def velocities(self):
        result = []
        for body in self.body_ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            result.append(velocity)
        return np.array(result)  # angular xyz, linear xyz, all in world coordinates

    def stone_energy(self):
        energy = 0.
        velocities = self.velocities()
        for i in np.flatnonzero(self.alive):
            body = self.body_ids[i]
            rotation = self.data.xmat[body].reshape(3, 3)
            local_omega = rotation.T @ velocities[i, :3]
            energy += .5 * self.model.body_mass[body] * np.dot(velocities[i, 3:], velocities[i, 3:])
            energy += .5 * np.dot(self.model.body_inertia[body], local_omega * local_omega)
        return float(energy)

    def _robot_target(self, t):
        command = self.command
        direction = np.array([math.cos(math.radians(command.angle)), math.sin(math.radians(command.angle))])
        duration = .70 / command.speed  # constant acceleration over a 0.35 m stroke
        if t < duration:
            distance = .35 * (t / duration) ** 2
            self.data.ctrl[:2] = direction * distance
            self.data.ctrl[2] = 0.
            self.data.ctrl[3] = .5 * command.omega * t * t / duration
            return
        if self.release is None:
            self.data.eq_active[self.grasps[self.current]] = 0
            self.phase = "SLIDING"
            self.release = float(self.data.time)
            self.last_release = {
                "position": self.positions()[self.current].tolist(),
                "velocity": self.velocities()[self.current].tolist(),
                "command_speed": command.speed, "command_omega": command.omega,
                "time": float(self.data.time), "stone_kinetic_energy": self.stone_energy(),
            }
        after = t - duration
        # Lift, brake, return above the start, then lower before the next shot.
        return_fraction = np.clip((after - .2) / .7, 0., 1.)
        smooth = return_fraction * return_fraction * (3 - 2 * return_fraction)
        self.data.ctrl[:2] = direction * .35 * (1 - smooth)
        if after < .2:
            self.data.ctrl[2] = .2 * after / .2
        elif after < .9:
            self.data.ctrl[2] = .2
        else:
            self.data.ctrl[2] = .2 * max(0., 1 - (after - .9) / .3)
        self.data.ctrl[3] = .5 * command.omega * duration * (1 - smooth)

    def step(self, sweeping=False):
        if self.phase not in ("DELIVERY", "SLIDING", "SETTLING"):
            return
        self._robot_target(float(self.data.time) - self.start_time)
        self.sweeping = bool(sweeping and self.release is not None)
        self.data.xfrc_applied[:] = 0.
        velocities = self.velocities()
        for i in range(3):
            swept = self.sweeping and i == self.current and self.alive[i]
            if not self.planar:
                self.model.pair_friction[self.ice_pairs[i], :2] = self.friction * (1 - .20 * swept)
            if not self.alive[i]:
                continue
            velocity = velocities[i, 3:5]
            speed = np.linalg.norm(velocity)
            if self.planar:
                # Assume a level sheet with N=mg. Friction is dissipative and
                # capped to avoid reversing a near-stationary stone in one step.
                mass = self.model.body_mass[self.body_ids[i]]
                mu = self.friction * (1 - .20 * swept)
                if speed > 1e-10:
                    deceleration = min(mu * 9.81, speed / DT)
                    self.data.xfrc_applied[self.body_ids[i], :2] -= mass * deceleration * velocity / speed
                omega = velocities[i, 2]
                inertia = self.model.body_inertia[self.body_ids[i], 2]
                torque = min(mu * mass * 9.81 * RADIUS * .03, inertia * abs(omega) / DT)
                self.data.xfrc_applied[self.body_ids[i], 5] = -math.copysign(torque, omega)
            if self.data.eq_active[self.grasps[i]]:
                continue
            if speed > .01:
                left = np.array([-velocity[1], velocity[0]]) / speed
                acceleration = self.curl * (1 - .30 * swept) * math.tanh(velocities[i, 2]) * speed / (speed + .1)
                self.data.xfrc_applied[self.body_ids[i], :2] += self.model.body_mass[self.body_ids[i]] * acceleration * left
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.disk_ids and b in self.disk_ids:
                self.collisions.add(tuple(sorted((self.disk_ids.index(a), self.disk_ids.index(b)))))
        positions, velocities = self.positions(), self.velocities()
        if self.release is not None:
            self.max_passive_energy = max(self.max_passive_energy, self.stone_energy())
        if self.alive.any():
            self.max_height = max(self.max_height, float(positions[self.alive, 2].max()))
            self.max_speed = max(self.max_speed, float(np.linalg.norm(velocities[self.alive, 3:], axis=1).max()))
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or self.max_height > .10 or self.max_speed > 4.):
            self.error, self.phase = "Unstable physical state", "ERROR"
            return
        for i in np.flatnonzero(self.alive):
            x, y = positions[i, :2]
            if x - RADIUS < 0 or x + RADIUS > 4 or abs(y) + RADIUS > .75:
                self.removed_positions[int(i)] = positions[i].tolist()
                self.alive[i] = False
                self.data.eq_active[self.grasps[i]] = 0
                q, v = self.qpos_addresses[i], self.qvel_addresses[i]
                if self.planar:
                    self.data.qpos[q:q + 3] = 0.
                    self.data.qvel[v:v + 3] = 0.
                else:
                    self.data.qpos[q:q + 7] = [-1., 2. + .3 * i, .026, 1., 0., 0., 0.]
                    self.data.qvel[v:v + 6] = 0.
                self._sync_visibility()
                mujoco.mj_forward(self.model, self.data)
        t = float(self.data.time) - self.start_time
        if self.release is not None:
            remaining = self.velocities()[self.alive]
            still = not len(remaining) or bool(
                np.all(np.linalg.norm(remaining[:, 3:], axis=1) < .01)
                and np.all(np.abs(remaining[:, 2]) < .05))
            robot_home = t - (self.release - self.start_time) > 1.6
            if still and robot_home:
                self.phase = "SETTLING"
                self.settle_time += DT
                if self.settle_time >= .4:
                    self.phase = "RESULT" if self.played.all() else "READY"
                    self.sweeping = False
            else:
                self.phase, self.settle_time = "SLIDING", 0.
        if t > 15 and self.phase not in ("READY", "RESULT"):
            self.error, self.phase = "Stone did not stop within 15 simulation seconds", "ERROR"

    def run_until_rest(self, sweeping=False, callback=None):
        steps = 0
        while self.phase in ("DELIVERY", "SLIDING", "SETTLING"):
            active_sweep = sweeping(self) if callable(sweeping) else sweeping
            self.step(active_sweep)
            if callback is not None and steps % 20 == 0:
                callback(self)
            steps += 1
        if self.phase == "ERROR":
            raise RuntimeError(self.error)
        return self.result()

    def result(self):
        positions = self.positions()
        stones = []
        for i in np.flatnonzero(self.played):
            stones.append({"id": int(i), "alive": bool(self.alive[i]),
                           "position": self.removed_positions.get(int(i), positions[i].tolist()),
                           "score": score_stone(positions[i], self.alive[i])})
        return {"phase": self.phase, "stones": stones, "score": sum(s["score"] for s in stones),
                "collisions": [list(pair) for pair in sorted(self.collisions)],
                "max_height": self.max_height, "max_speed": self.max_speed,
                "max_passive_energy": self.max_passive_energy, "planar": self.planar,
                "release": copy.deepcopy(self.last_release), "simulation_time": float(self.data.time)}

    def observe_board(self):
        """Export only planar kinematics and already executed control events."""
        if not self.planar or self.phase == "ERROR":
            raise ValueError("Board observations require a valid planar world.")
        positions, velocities = self.positions(), self.velocities()
        stones = []
        for i in np.flatnonzero(self.played):
            position = self.removed_positions.get(int(i), positions[i])
            stones.append(StoneObservation(int(i), bool(self.alive[i]), tuple(map(float, position[:2])),
                                          tuple(map(float, velocities[i, 3:5])),
                                          float(self.data.qpos[self.qpos_addresses[i] + 2]), float(velocities[i, 2])))
        return BoardObservation(float(self.data.time), self.phase, tuple(stones),
                                tuple(map(float, self.data.qpos[:4])), tuple(map(float, self.data.qvel[:4])),
                                self.current, self.command, float(self.start_time), self.release, float(self.settle_time))

    def restore_observation(self, observation: BoardObservation):
        """Restore measurements in a private forecast world using its own ice.

        Unlike restore(snapshot), this never copies the source world's forces
        or solver state. The standard scene's first four joints are the robot.
        """
        if not self.planar:
            raise ValueError("Observation restoration supports the planar game only.")
        if observation.phase not in ("READY", "DELIVERY", "SLIDING", "SETTLING", "RESULT"):
            raise ValueError("Unsupported observed game phase.")
        scalars = [observation.time, observation.start_time, observation.settle_time]
        if observation.release_time is not None:
            scalars.append(observation.release_time)
        if (not np.isfinite(scalars).all() or observation.time < 0 or observation.start_time < 0
                or observation.start_time > observation.time or observation.settle_time < 0
                or len(observation.robot_position) != 4 or len(observation.robot_velocity) != 4
                or not np.isfinite(observation.robot_position).all() or not np.isfinite(observation.robot_velocity).all()):
            raise ValueError("Invalid observed timing or robot kinematics.")
        ids = [s.id for s in observation.stones]
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 3 for i in ids):
            raise ValueError("Observed stone IDs must be unique integers in [0, 2].")
        active = observation.phase in ("DELIVERY", "SLIDING", "SETTLING")
        if active and (observation.current not in ids or observation.command is None):
            raise ValueError("An active shot requires its stone ID and executed command.")
        if observation.release_time is not None and not observation.start_time <= observation.release_time <= observation.time:
            raise ValueError("Invalid release event time.")
        if observation.phase in ("SLIDING", "SETTLING") and observation.release_time is None:
            raise ValueError("A gliding stone requires a release event.")
        if observation.phase == "DELIVERY" and (observation.release_time is not None
                or not next(s for s in observation.stones if s.id == observation.current).alive):
            raise ValueError("Delivery must hold a live, unreleased stone.")
        for stone in observation.stones:
            if (len(stone.position) != 2 or len(stone.velocity) != 2
                    or not np.isfinite([*stone.position, *stone.velocity, stone.yaw, stone.angular_velocity]).all()):
                raise ValueError("Invalid stone kinematics.")
            x, y = stone.position
            if stone.alive and (x < RADIUS - 1e-9 or x > 4 - RADIUS + 1e-9 or abs(y) > .75 - RADIUS + 1e-9):
                raise ValueError("A live stone must be inside the rink.")
        self.reset()
        self.data.time = observation.time
        self.data.qpos[:4] = observation.robot_position
        self.data.qvel[:4] = observation.robot_velocity
        for stone in observation.stones:
            self.played[stone.id] = True
            if stone.alive:
                self._place(stone.id, stone.position)
                self.data.qpos[self.qpos_addresses[stone.id] + 2] = stone.yaw
                address = self.qvel_addresses[stone.id]
                self.data.qvel[address:address + 3] = [*stone.velocity, stone.angular_velocity]
            else:
                self.removed_positions[stone.id] = [*stone.position, .026]
        self.phase, self.command, self.current = observation.phase, observation.command, observation.current
        self.start_time, self.release, self.settle_time = observation.start_time, observation.release_time, observation.settle_time
        if self.phase == "DELIVERY":
            self.data.eq_active[self.grasps[self.current]] = 1
        # _place may have recomputed dynamics with earlier stones only. Start
        # all solver history fresh, and recompute from the final observed pose.
        self.data.qacc_warmstart[:] = 0.
        self.data.qfrc_applied[:] = 0.
        self.data.xfrc_applied[:] = 0.
        self._sync_visibility()
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self):
        state = np.empty(mujoco.mj_stateSize(self.model, STATE_SIGNATURE))
        mujoco.mj_getState(self.model, self.data, state, STATE_SIGNATURE)
        fields = ("played", "alive", "removed_positions", "phase", "command", "current", "start_time",
                  "settle_time", "release", "last_release", "collisions", "max_height", "max_speed", "max_passive_energy", "sweeping", "error")
        return {"state": state, "planar": self.planar,
                "fields": {key: copy.deepcopy(getattr(self, key)) for key in fields}}

    def restore(self, snapshot):
        if snapshot["planar"] != self.planar:
            raise ValueError("Snapshot and physics model have different motion constraints.")
        mujoco.mj_setState(self.model, self.data, snapshot["state"], STATE_SIGNATURE)
        for key, value in snapshot["fields"].items():
            setattr(self, key, copy.deepcopy(value))
        self._sync_visibility()
        mujoco.mj_forward(self.model, self.data)


def reference_forecast(snapshot, command, friction, curl, planar=True):
    """Known-ice reference forecast for the technical PoC, NOT learned AI."""
    world = CurlingWorld(friction, curl, planar=planar)
    world.restore(snapshot)
    if world.phase == "READY":
        world.start_shot(command)
    trace = []
    result = world.run_until_rest(callback=lambda w: trace.append(w.positions().tolist()))
    return result, trace
