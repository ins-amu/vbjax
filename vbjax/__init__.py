from .loops import make_sde, make_ode, make_dde, make_sdde
from .shtlc import make_shtdiff
from .neural_mass import JRState, JRTheta, jr_dfun, jr_default_theta
from .regmap import make_region_mapping
from ._version import __version__