"""Render a demo episode: global map GIF + egocentric observation strip."""

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from sim import WarehouseSim
from collect import smart_random_policy


def run_demo(seed=3, steps=160, gif_path="warehouse_demo.gif",
             frame_path="warehouse_frames.png"):
    sim = WarehouseSim(seed=seed, obs_size=96)
    obs, info = sim.reset()
    rng = np.random.default_rng(seed)

    global_frames, ego_frames, collided = [], [], False
    for t in range(steps):
        a = smart_random_policy(sim, rng, collided)
        obs, r, term, trunc, info = sim.step(a)
        collided = info["collision"] is not None
        if t % 2 == 0:
            global_frames.append(sim.render_global())
        if t % 20 == 0:
            ego_frames.append(obs)
        if term:
            print(f"reached the goal at t={t}")
            break

    imageio.mimsave(gif_path, global_frames, fps=12, loop=0)
    print(f"saved {gif_path} ({len(global_frames)} frames)")

    # side-by-side sheet: global view + strip of egocentric observations
    g = Image.fromarray(sim.render_global()).resize((360, 360), Image.NEAREST)
    strip = Image.new("RGB", (360, 360 + 8 + 120), (245, 245, 245))
    strip.paste(g, (0, 0))
    for i, e in enumerate(ego_frames[:3]):
        im = Image.fromarray(e).resize((116, 116), Image.NEAREST)
        strip.paste(im, (i * 122, 368))
    strip.save(frame_path)
    print(f"saved {frame_path}")


if __name__ == "__main__":
    run_demo()
