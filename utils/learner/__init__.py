# taken from https://github.com/taeckyung/SoTTA/tree/main/learner
from .learner import make, register

from . import sam_adapt
from . import sar
from . import tent
from . import spa
from . import foa
from . import norm  # BP-free adaptation via BatchNorm recalibration
from . import foa_simple  # BP-free activation shifting for ViT
from . import lame  # BP-free Laplacian-adjusted MLE (Boudiaf et al. CVPR 2022)
from . import pea   # BP-free Progressive Embedding Alignment (Xiao et al. ICLR 2026)
from . import no_adapt     # Passthrough lower bound
from . import oracle       # Supervised Oracle for SAR
from . import oracle_spa   # Supervised Oracle for SPA
from . import ga_gated_sar # Gradient-alignment-gated SAR (diagnostic)
from . import sar_correct  # Correctness-filtered SAR (diagnostic)
