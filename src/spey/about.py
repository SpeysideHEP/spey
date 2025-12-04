"""Function to display details about the spey installation"""

import platform
import sys
from importlib.metadata import distribution, version
from importlib.util import find_spec
from subprocess import check_output


def about() -> None:
    """Prints the information regarding spey installation"""

    print(check_output([sys.executable, "-m", "pip", "show", "spey"]).decode())
    print(f"Platform info:            {platform.platform(aliased=True)}")
    print(
        f"Python version:           {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version:            {version('numpy')}")
    print(f"Scipy version:            {version('scipy')}")
    if find_spec("iminuit") is not None:
        print(f"iminuit version:          {version('iminuit')}")
    print(f"Autograd version:         {version('autograd')}")
    print(f"tqdm version:             {version('tqdm')}")
    print(f"semantic_version version: {version('semantic_version')}")

    print("\nInstalled backend plug-ins:\n")

    shown = ["spey"]
    from spey import _get_entry_points

    plugin_devices = _get_entry_points("spey.backend.plugins")
    for d in plugin_devices:
        try:
            dist_name = d.dist.name
            dist_version = d.dist.version
        except AttributeError:
            dist_name = d.value.split(":")[0].split(".")[0]
            dist_version = distribution(dist_name).version
        print(f"- {d.name} ({dist_name}-{dist_version})")
        if dist_name not in shown:
            print(
                check_output(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "show",
                        d.dist.metadata.json["name"],
                    ]
                ).decode()
            )
            shown.append(d.dist.metadata.json["name"])
