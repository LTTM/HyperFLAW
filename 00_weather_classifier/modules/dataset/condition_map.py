# Define the export
__all__ = ['condition_map']


# Constants
MAPPING_WEATHER = {"clear": 0, "night": 1, "rain": 2, "fog": 3}
MAPPING_FULL = {
    "car_clear": 0,
    "car_night": 1,
    "car_rain": 2,
    "car_fog": 3,
    "drone_clear": 4,
    "drone_night": 5,
    "drone_rain": 6,
    "drone_fog": 7,
}
REVERSE_MAPPING_WEATHER = {v: k for k, v in MAPPING_WEATHER.items()}
FULL2WEATHER = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 0,
    5: 1,
    6: 2,
    7: 3,
}


def condition_map(condition: list or str) -> list:
    if isinstance(condition, str):
        condition = [condition]
    out = [0 for _ in range(len(condition))]

    for i, elm in enumerate(condition):
        try:
            out[i] = FULL2WEATHER[MAPPING_FULL[elm]]
        except KeyError:
            out[i] = FULL2WEATHER[MAPPING_WEATHER[elm]]

    return out if len(out) > 1 else out[0]
