"""Dashboard sub-package for energy market EDA and strategy evaluation."""

from __future__ import annotations

import re


def class_display_name(cls: type) -> str:
    """Convert a CamelCase strategy class name to a human-readable display name.

    E.g. ``NaiveCopyStrategy`` -> ``Naive Copy``,
         ``PerfectForesightStrategy`` -> ``Perfect Foresight``.
    """
    name = cls.__name__
    name = re.sub(r"Strategy$", "", name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.strip()
