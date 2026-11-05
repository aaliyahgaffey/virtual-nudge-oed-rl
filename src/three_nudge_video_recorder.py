"""Capture and replay an actual MuJoCo simulation as an MP4.

Used with three_nudge_oed_v2_visual.run_mujoco(..., recorder=recorder).
Only saved MuJoCo states are replayed. This does not emulate or recreate
physical pad contact: the nudge force in the experiment is virtual.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Snapshot:
    t: float
    qpos: np.ndarray
    qvel: np.ndarray
    stage: str
    cbf_interventions: int

class NudgeRecorder:
    def __init__(self, capture_fps=10.0):
        if capture_fps <= 0:
            raise ValueError('capture_fps must be positive')
        self.capture_fps = float(capture_fps)
        self.snapshots = []
        self.plans = {}
        self._next_capture = None

    def capture(self, experiment, stage, force=False):
        now = float(experiment.data.time)
        if self._next_capture is None:
            self._next_capture = now
        if not force and now + 1e-10 < self._next_capture:
            return
        self.snapshots.append(Snapshot(
            now,
            np.array(experiment.data.qpos, dtype=float).copy(),
            np.array(experiment.data.qvel, dtype=float).copy(),
            str(stage), int(experiment.cbf_interventions_total)))
        self._next_capture = now + 1.0/self.capture_fps

    def plan(self, number, cand, observation_time_s):
        self.plans[int(number)] = dict(
            point_body=np.array(cand.point_body, dtype=float).copy(),
            push_body=np.array(cand.push_body, dtype=float).copy(),
            J=float(cand.J),F=float(cand.force_N),
            T=float(observation_time_s))

    def save_npz(self, filename):
        if not self.snapshots:
            raise RuntimeError('No snapshots to save. Run the experiment with recorder=recorder.')
        path=Path(filename)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,
            times=np.array([s.t for s in self.snapshots]),
            qpos=np.array([s.qpos for s in self.snapshots]),
            qvel=np.array([s.qvel for s in self.snapshots]),
            stages=np.array([s.stage for s in self.snapshots]),
            cbf_interventions=np.array([s.cbf_interventions for s in self.snapshots]))
        return path

    def export_mp4(self, model, output_path, base_body_id, target_body_id,
                   width=960,height=540,output_fps=15, camera_distance=7.7,
                   azimuth=130,elevation=-20,show_cbf_proxies=False):
        """Render states with MuJoCo Renderer; requires functional GL backend.

        output_fps determines *playback speed* because one captured snapshot
        produces one video frame. Capture at 10 fps and export at 15 fps
        displays the motion 1.5x faster than real simulation time.
        """
        import mujoco
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        if not self.snapshots:
            raise RuntimeError('No snapshots captured; run the MuJoCo experiment first.')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        # Default MuJoCo XML often has a 640 x 480 offscreen framebuffer.
        # Avoid requesting more pixels than the actual model supports.
        width = min(int(width), int(model.vis.global_.offwidth))
        height = min(int(height), int(model.vis.global_.offheight))
        if width % 2: width -= 1
        if height % 2: height -= 1
        d=mujoco.MjData(model)
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance=float(camera_distance)
        cam.azimuth=float(azimuth)
        cam.elevation=float(elevation)
        opt=mujoco.MjvOption()
        opt.geomgroup[5] = 1 if show_cbf_proxies else 0
        writer=None
        font=ImageFont.load_default()
        try:
            with mujoco.Renderer(model,height=height,width=width) as renderer:
                writer=imageio.get_writer(str(output_path), fps=output_fps,
                        codec='libx264', quality=7, macro_block_size=1,
                        ffmpeg_log_level='error')
                for s in self.snapshots:
                    d.qpos[:]=s.qpos
                    d.qvel[:]=s.qvel
                    d.time=s.t
                    mujoco.mj_forward(model,d)
                    base=np.asarray(d.xpos[base_body_id])
                    target=np.asarray(d.xpos[target_body_id])
                    cam.lookat[:]=(base+target)/2.0
                    renderer.update_scene(d,camera=cam,scene_option=opt)
                    number=None
                    chunks=s.stage.split()
                    if len(chunks)==2 and chunks[0] in ('push','observe'):
                        try: number=int(chunks[1])
                        except ValueError:pass
                    # A thin, colored arrow illustrates the intended external
                    # push direction. It is NOT a physical arm or contact.
                    if number in self.plans and chunks[0]=='push':
                        plan=self.plans[number]
                        if renderer.scene.ngeom < renderer.scene.maxgeom:
                            geom=renderer.scene.geoms[renderer.scene.ngeom]
                            R=d.xmat[target_body_id].reshape(3,3)
                            start=d.xpos[target_body_id]+R@plan['point_body']
                            finish=start+0.55*(R@plan['push_body'])
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,
                                               np.zeros(3),np.zeros(3),np.eye(3).ravel(),
                                               np.array([1.0,0.2,0.06,1.0],dtype=np.float32))
                            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,
                                                0.055,start,finish)
                            geom.category=mujoco.mjtCatBit.mjCAT_DECOR
                            renderer.scene.ngeom+=1
                    frame=renderer.render()
                    image=Image.fromarray(frame).convert('RGBA')
                    overlay=Image.new('RGBA',image.size,(0,0,0,0))
                    draw=ImageDraw.Draw(overlay)
                    draw.rectangle([0,0,width,73],fill=(0,0,0,165))
                    label=s.stage.title()
                    info=f'  |  time {s.t:.2f} s  |  CBF QP interventions: {s.cbf_interventions}'
                    draw.text((16,12),'THREE-NUDGE OED  |  '+label+info,
                              fill=(255,255,255,255),font=font)
                    if number in self.plans:
                        plan=self.plans[number]
                        caption=(f'Impulse {plan["J"]:.3f} N s  |  Force {plan["F"]:.3f} N'
                                 f'  |  Planned observation {plan["T"]:.0f} s')
                        draw.text((16,34),caption,fill=(230,233,248,255),font=font)
                    else:
                        draw.text((16,34),'Red arrow = virtual target push (not physical arm contact)',
                                  fill=(230,233,248,255),font=font)
                    image=Image.alpha_composite(image,overlay).convert('RGB')
                    writer.append_data(np.asarray(image))
        finally:
            if writer is not None:
                writer.close()
        return output_path
