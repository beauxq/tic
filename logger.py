from typing import Any

import numpy as np


class Logger:
    def __init__(self):
        self.logging = False

    def log(self, *out: list[float] | np.ndarray[Any, Any] | Any) -> None:
        """ print, only if logging is true """
        if self.logging:
            if (len(out) == 1) \
                    and isinstance(out[0], (list, np.ndarray)) \
                    and len(out[0]) == 9:
                # guess it's a tic tac toe board
                for i, value in enumerate(out[0]):
                    if isinstance(value, float):
                        value = round(value, 3)
                    elif value > -1:
                        value = " " + str(value)
                    print(value, end=("\n" if i % 3 == 2 else " "))
                print()
            else:
                print(*out)
