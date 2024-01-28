"""Function to display details about the spey installation"""

import platform
import sys
from subprocess import check_output

import numpy
import scipy
import semantic_version
import tqdm
from pkg_resources import get_distribution, iter_entry_points


def about() -> None:
    """Prints the information regarding spey installation"""

    print(check_output([sys.executable, "-m", "pip", "show", "spey"]).decode())
    print(f"Platform info:            {platform.platform(aliased=True)}")
    print(
        f"Python version:           {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version:            {numpy.__version__}")
    print(f"Scipy version:            {scipy.__version__}")
    print(f"Autograd version:         {get_distribution('autograd').version}")
    print(f"tqdm version:             {tqdm.__version__}")
    print(f"semantic_version version: {semantic_version.__version__}")

    print("\nInstalled backend plug-ins:\n")

    shown = ["spey"]
    plugin_devices = iter_entry_points("spey.backend.plugins")
    for d in plugin_devices:
        print(f"- {d.name} ({d.dist.project_name}-{d.dist.version})")
        if d.dist.project_name not in shown:
            print(
                check_output(
                    [sys.executable, "-m", "pip", "show", d.dist.project_name]
                ).decode()
            )
            shown.append(d.dist.project_name)
