import importlib
import json
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path

import torch

for module in (
    'lerobot', 'torchvision', 'av', 'datasets', 'serial', 'rustypot',
    'rerun', 'foxglove', 'lerobot.motors.motors_bus',
    'lerobot.teleoperators.so_leader', 'lerobot.robots.so_follower',
):
    importlib.import_module(module)
    print(f'IMPORT OK: {module}', flush=True)

print(json.dumps({name: metadata.version(name) for name in (
    'lerobot', 'torch', 'torchvision', 'torchcodec', 'rustypot', 'av',
)}, indent=2), flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T)
torch.cuda.synchronize()
print(f'CUDA COMPUTE OK: {torch.cuda.get_device_name(0)}; runtime {torch.version.cuda}', flush=True)

from torchcodec.decoders import VideoDecoder

with tempfile.TemporaryDirectory(prefix='lerobot-verify-') as directory:
    path = Path(directory) / 'test.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc=size=64x64:rate=5', '-t', '1', '-pix_fmt', 'yuv420p',
                    str(path)], check=True)
    decoder = VideoDecoder(str(path), device='cpu')
    assert decoder[0].shape == (3, 64, 64)
    print(f'VIDEO ENCODE/DECODE OK: {len(decoder)} frames', flush=True)

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_string('''<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1" rgba=".5 .5 .5 1"/>
    <body pos="0 0 .5"><freejoint/>
      <geom type="box" size=".025 .025 .025" mass=".02" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>''')
data = mujoco.MjData(model)
for _ in range(500):
    mujoco.mj_step(model, data)
assert .01 < data.qpos[2] < .04, data.qpos
with mujoco.Renderer(model, height=240, width=320) as renderer:
    renderer.update_scene(data)
    pixels = renderer.render()
    assert pixels.shape == (240, 320, 3)
    assert np.std(pixels) > 1
    from OpenGL import GL
    print('OPENGL RENDERER:', GL.glGetString(GL.GL_RENDERER), flush=True)
print(f'MUJOCO PHYSICS AND EGL RENDER OK: {mujoco.__version__}', flush=True)
print('INSTALLATION VERIFICATION PASSED', flush=True)
