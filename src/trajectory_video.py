"""
trajectory_video.py
===================

Notebook-friendly recording and MP4 rendering utilities for the
active_inertial_id MuJoCo experiment.

This module records full MuJoCo qpos/qvel while the EXISTING experiment code
runs.  It does not change the controller, CBF, nudge logic, or estimator.

Typical notebook workflow
-------------------------
    from trajectory_video import (
        TrajectoryRecorder,
        render_trajectory_video,
        make_nudge_observation_segments,
    )

    with TrajectoryRecorder(exp, sample_period_s=0.05) as recorder:
        nudge_result = exp.run_nudge(verbose=True)

        observation_start_s = float(exp.data.time)
        post_log = exp.observe_free_flight(
            config.post_nudge_observation_s,
            config.post_nudge_sample_period_s,
            progress_every_s=120.0,
        )
        observation_end_s = float(exp.data.time)

    trajectory = recorder.log()

    segments = make_nudge_observation_segments(
        trajectory,
        nudge_result,
        observation_start_s=observation_start_s,
        observation_end_s=observation_end_s,
        observation_speedup=30.0,
    )

    video_path = render_trajectory_video(
        exp.model,
        trajectory,
        "videos/nudge_and_observation.mp4",
        fps=30,
        segments=segments,
        show_cbf=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import mujoco


CBF_GROUP = 5


@dataclass
class TrajectoryLog:
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times_s=self.times_s,
            qpos=self.qpos,
            qvel=self.qvel,
        )
        return path

    @classmethod
    def load(cls, path) -> "TrajectoryLog":
        d = np.load(path)
        return cls(
            times_s=np.asarray(d["times_s"], dtype=float),
            qpos=np.asarray(d["qpos"], dtype=float),
            qvel=np.asarray(d["qvel"], dtype=float),
        )


class TrajectoryRecorder:
    """
    Record full MuJoCo state by wrapping exp.execute_step().

    The recorder is deliberately external to SingleNudgeExperiment so your
    scientific/controller code stays unchanged.

    sample_period_s=0.05 records at 20 Hz in simulation time.  For a 1200 s
    observation this is about 24,000 states, which is normally quite
    manageable and is more than enough for later accelerated video rendering.
    """

    def __init__(
        self,
        experiment,
        sample_period_s: float = 0.05,
    ):
        if sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive.")

        self.exp = experiment
        self.sample_period_s = float(sample_period_s)

        self._times = []
        self._qpos = []
        self._qvel = []

        self._original_execute_step = None
        self._next_capture_t = None
        self._running = False

    def _capture(self) -> None:
        self._times.append(float(self.exp.data.time))
        self._qpos.append(
            np.asarray(self.exp.data.qpos, dtype=float).copy()
        )
        self._qvel.append(
            np.asarray(self.exp.data.qvel, dtype=float).copy()
        )

    def _capture_if_due(self) -> None:
        t = float(self.exp.data.time)

        if self._next_capture_t is None:
            self._next_capture_t = t

        if t + 1e-12 >= self._next_capture_t:
            self._capture()

            # Move the next requested sample time forward until it is in the
            # future.  This remains robust if the simulation timestep changes.
            while self._next_capture_t <= t + 1e-12:
                self._next_capture_t += self.sample_period_s

    def start(self):
        if self._running:
            return self

        self._times.clear()
        self._qpos.clear()
        self._qvel.clear()

        self._original_execute_step = self.exp.execute_step
        self._next_capture_t = float(self.exp.data.time)

        self._capture()
        self._next_capture_t += self.sample_period_s

        original = self._original_execute_step

        def wrapped_execute_step(
            u_nom,
            accumulate_impulse=False,
        ):
            result = original(
                u_nom,
                accumulate_impulse=accumulate_impulse,
            )
            self._capture_if_due()
            return result

        # Assigning the closure directly to the instance is intentional:
        # calls such as exp.execute_step(u) then invoke this wrapper without
        # Python adding an extra bound-method self argument.
        self.exp.execute_step = wrapped_execute_step
        self._running = True
        return self

    def stop(self):
        if not self._running:
            return

        # Capture the final state even if it falls between scheduled samples.
        if (
            not self._times
            or abs(float(self.exp.data.time) - self._times[-1]) > 1e-12
        ):
            self._capture()

        self.exp.execute_step = self._original_execute_step
        self._running = False

    def log(self) -> TrajectoryLog:
        if not self._times:
            raise RuntimeError(
                "No trajectory has been recorded yet."
            )

        return TrajectoryLog(
            times_s=np.asarray(self._times, dtype=float),
            qpos=np.asarray(self._qpos, dtype=float),
            qvel=np.asarray(self._qvel, dtype=float),
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def make_nudge_observation_segments(
    trajectory: TrajectoryLog,
    nudge_result,
    observation_start_s: float,
    observation_end_s: Optional[float] = None,
    *,
    contact_force_threshold_N: float = 0.10,
    pre_nudge_speedup: float = 20.0,
    approach_speedup: float = 10.0,
    contact_speedup: float = 1.0,
    post_contact_speedup: float = 5.0,
    observation_speedup: float = 30.0,
    contact_padding_s: float = 1.0,
    retreat_speedup: Optional[float] = None,
):
    """
    Build time-warp segments that keep the actual contact easy to see while
    accelerating the long observation.

    Returns a list of:
        (segment_start_sim_time, segment_end_sim_time, playback_speedup)

    In the self-collision-only experiment there is no commanded target-relative
    retreat. ``post_contact_speedup`` controls only the short interval between
    the end of the padded contact window and the start of the long observation.

    ``retreat_speedup`` is retained as a backwards-compatible alias for older
    notebooks. If supplied, it overrides ``post_contact_speedup``.

    Example:
      - pre-nudge baseline:       20x
      - approach:                 10x
      - contact:                   1x
      - post-contact transition:   5x
      - 1200 s observation:       30x
    """
    if retreat_speedup is not None:
        post_contact_speedup = float(retreat_speedup)
    t0 = float(trajectory.times_s[0])
    tf = (
        float(trajectory.times_s[-1])
        if observation_end_s is None
        else float(observation_end_s)
    )

    force_t = np.asarray(
        nudge_result.force_time_history_s,
        dtype=float,
    )
    force = np.asarray(
        nudge_result.force_push_history_N,
        dtype=float,
    )

    if force_t.size == 0:
        raise ValueError(
            "nudge_result.force_time_history_s is empty."
        )

    approach_start = float(force_t[0])

    contact_idx = np.flatnonzero(
        force > float(contact_force_threshold_N)
    )

    if contact_idx.size:
        contact_start = max(
            approach_start,
            float(force_t[contact_idx[0]]) - contact_padding_s,
        )
        contact_end = min(
            float(observation_start_s),
            float(force_t[contact_idx[-1]]) + contact_padding_s,
        )
    else:
        # If the force threshold does not identify contact, retain a short
        # high-detail window at the end of the nudge loop.
        contact_end = min(
            float(observation_start_s),
            float(force_t[-1]),
        )
        contact_start = max(
            approach_start,
            contact_end - 5.0,
        )

    obs_start = float(observation_start_s)
    obs_end = min(tf, float(observation_end_s or tf))

    raw = [
        (t0, approach_start, pre_nudge_speedup),
        (approach_start, contact_start, approach_speedup),
        (contact_start, contact_end, contact_speedup),
        (contact_end, obs_start, post_contact_speedup),
        (obs_start, obs_end, observation_speedup),
    ]

    segments = []
    for a, b, speed in raw:
        a = max(t0, float(a))
        b = min(tf, float(b))
        if b > a + 1e-9:
            if speed <= 0.0:
                raise ValueError(
                    "All playback speedups must be positive."
                )
            segments.append((a, b, float(speed)))

    return segments


def _frame_times_from_segments(
    trajectory: TrajectoryLog,
    fps: int,
    speedup: float,
    segments,
) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if speedup <= 0.0:
        raise ValueError("speedup must be positive.")

    t0 = float(trajectory.times_s[0])
    tf = float(trajectory.times_s[-1])

    if segments is None:
        segments = [(t0, tf, float(speedup))]

    frame_times = []

    for seg_i, (a, b, s) in enumerate(segments):
        a = max(t0, float(a))
        b = min(tf, float(b))
        s = float(s)

        if b <= a:
            continue
        if s <= 0.0:
            raise ValueError(
                "Segment speedups must be positive."
            )

        sim_dt_per_frame = s / float(fps)

        # arange deliberately excludes b to avoid duplicate frames at segment
        # boundaries.  The final trajectory state is appended separately.
        seg_times = np.arange(
            a,
            b,
            sim_dt_per_frame,
            dtype=float,
        )

        if seg_times.size:
            frame_times.extend(seg_times.tolist())

    if not frame_times:
        frame_times = [t0]

    if frame_times[-1] < tf - 1e-9:
        frame_times.append(tf)

    return np.asarray(frame_times, dtype=float)


def _nearest_state_indices(
    recorded_times: np.ndarray,
    requested_times: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(
        recorded_times,
        requested_times,
        side="left",
    )

    idx = np.clip(
        idx,
        0,
        len(recorded_times) - 1,
    )

    prev = np.clip(
        idx - 1,
        0,
        len(recorded_times) - 1,
    )

    choose_prev = (
        np.abs(requested_times - recorded_times[prev])
        <= np.abs(recorded_times[idx] - requested_times)
    )

    return np.where(
        choose_prev,
        prev,
        idx,
    )


def render_trajectory_video(
    model: mujoco.MjModel,
    trajectory: TrajectoryLog,
    output_path,
    *,
    fps: int = 30,
    speedup: float = 30.0,
    segments=None,
    width: int = 600,
    height: int = 400,
    show_cbf: bool = True,
    camera_mode: str = "follow_pair",
    # azimuth_deg: float = 135.0,
    azimuth_deg: float = 90.0,
    # elevation_deg: float = -20.0,
    elevation_deg: float = 0.0,
    minimum_camera_distance_m: float = 6.0,
    camera_distance_scale: float = 1.25,
    camera_distance_padding_m: float = 3.0,
) -> Path:
    """
    Render the recorded trajectory to an MP4.

    Parameters
    ----------
    speedup:
        Global playback speed if `segments` is None.
        Example: speedup=30 makes 1200 simulation seconds play in ~40 seconds.

    segments:
        Optional list of (start_sim_s, end_sim_s, speedup).  This allows the
        contact event to play slowly while the long observation plays quickly.

    camera_mode:
        "follow_pair" dynamically frames the servicer and asteroid together.
        "fixed" leaves the camera centered near the servicer.

    show_cbf:
        True shows geom group 5 (the translucent CBF ellipsoids).
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Video rendering requires imageio and imageio-ffmpeg.\n"
            "Install with:\n"
            "  pip install imageio imageio-ffmpeg"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if trajectory.qpos.ndim != 2:
        raise ValueError(
            "trajectory.qpos must be a 2-D array."
        )

    frame_times = _frame_times_from_segments(
        trajectory,
        fps=fps,
        speedup=speedup,
        segments=segments,
    )

    state_indices = _nearest_state_indices(
        trajectory.times_s,
        frame_times,
    )

    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "spacecraft",
    )
    target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "asteroid",
    )

    if min(base_id, target_id) < 0:
        raise RuntimeError(
            "Expected bodies 'spacecraft' and 'asteroid' "
            "were not found in the model."
        )

    vis = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vis)
    vis.geomgroup[:] = 1
    vis.geomgroup[CBF_GROUP] = 1 if show_cbf else 0

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = float(azimuth_deg)
    cam.elevation = float(elevation_deg)

    renderer = mujoco.Renderer(
        model,
        height=int(height),
        width=int(width),
    )

    writer = imageio.get_writer(
        str(output_path),
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        n_frames = len(state_indices)

        for frame_no, state_idx in enumerate(state_indices):
            data.qpos[:] = trajectory.qpos[state_idx]
            data.qvel[:] = trajectory.qvel[state_idx]
            data.time = float(
                trajectory.times_s[state_idx]
            )

            mujoco.mj_forward(model, data)

            if camera_mode == "follow_pair":
                base_pos = np.asarray(
                    data.xpos[base_id],
                    dtype=float,
                )
                target_pos = np.asarray(
                    data.xpos[target_id],
                    dtype=float,
                )

                midpoint = 0.5 * (
                    base_pos + target_pos
                )
                separation = float(
                    np.linalg.norm(
                        target_pos - base_pos
                    )
                )

                cam.lookat[:] = midpoint
                cam.distance = max(
                    float(minimum_camera_distance_m),
                    float(camera_distance_scale)
                    * separation
                    + float(camera_distance_padding_m),
                )

            elif camera_mode == "fixed":
                cam.lookat[:] = [1.0, 0.0, 0.0]
                cam.distance = float(
                    minimum_camera_distance_m
                )

            else:
                raise ValueError(
                    "camera_mode must be "
                    "'follow_pair' or 'fixed'."
                )

            renderer.update_scene(
                data,
                camera=cam,
                scene_option=vis,
            )
            frame = renderer.render()
            writer.append_data(frame)

            if (
                frame_no == 0
                or (frame_no + 1) % max(1, fps * 5) == 0
                or frame_no + 1 == n_frames
            ):
                print(
                    f"\rRendering frame "
                    f"{frame_no + 1}/{n_frames}",
                    end="",
                    flush=True,
                )

    finally:
        writer.close()
        renderer.close()

    print(f"\nSaved video: {output_path}")
    return output_path
