"""script settings that an environment variable can override, so a container or cloud job can change
them without editing code. the default stays at the top of the script:

  START = setting("START", "2025-07-09T18:00")    # START=2025-08-01 in the environment overrides it
  PUSH = setting("PUSH", True)                     # true/false, 1/0 or yes/no

the environment variable has the setting's name. the value is converted to the default's type
(bool, int, float or str); an empty variable counts as unset.
"""
import os

TRUE, FALSE = {"true", "1", "yes"}, {"false", "0", "no"}


def setting(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    if isinstance(default, bool):
        if value.lower() not in TRUE | FALSE:
            raise ValueError(f"{name}={value!r} is not true or false")
        return value.lower() in TRUE
    if isinstance(default, (int, float)):
        return type(default)(value)
    return value
