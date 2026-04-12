from pathlib import Path
import sys
import os

# -------------------------
# Make repo root importable
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional: force current working directory to repo root
os.chdir(ROOT)

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.config import Config
from mmseg.registry import MODELS

CFG_PATH = ROOT / 'configs' / 'powerline_v1' / 'powerline_v1_r18_fpn.py'

cfg = Config.fromfile(str(CFG_PATH))
model = MODELS.build(cfg.model)

print('Repo root =', ROOT)
print('Config =', CFG_PATH)
print('Model class =', type(model))
print(model)
print('Model build passed.')