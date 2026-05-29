"""Function to display details about the spey installation"""

import platform
import sys
from importlib.metadata import distribution, version
from importlib.util import find_spec


def about() -> None:
    """Prints the information regarding spey installation"""

    from spey._version import __version__

    spey_meta = distribution("spey").metadata
    sep = "=" * 62

    print(sep)
    print(f"  spey v{__version__}")
    print(f"  {spey_meta['Summary']}")
    print(f"  {spey_meta['Project-URL']}")
    print(sep)

    print("\nSystem:")
    print(f"  Platform:             {platform.platform(aliased=True)}")
    print(
        f"  Python:               "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    print("\nCore dependencies:")
    _col = 22
    for pkg in ("numpy", "scipy", "autograd", "tqdm", "joblib", "semantic_version"):
        print(f"  {pkg:<{_col}}{version(pkg)}")
    if find_spec("iminuit") is not None:
        print(f"  {'iminuit':<{_col}}{version('iminuit')}  (optional)")

    print("\nInstalled backends:")

    from spey import _get_entry_points

    by_dist = {}
    for ep in _get_entry_points("spey.backend.plugins"):
        try:
            dist_name = ep.dist.name
            dist_ver = ep.dist.version
        except AttributeError:
            dist_name = ep.value.split(":")[0].split(".")[0]
            dist_ver = distribution(dist_name).version
        if dist_name not in by_dist:
            by_dist[dist_name] = (dist_ver, [])
        by_dist[dist_name][1].append(ep.name)

    for dist_name in sorted(by_dist):
        dist_ver, ep_names = by_dist[dist_name]
        print(f"\n  {dist_name} v{dist_ver}:")
        for ep_name in sorted(ep_names):
            print(f"    - {ep_name}")
