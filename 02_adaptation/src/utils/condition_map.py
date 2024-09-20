from utils.args import CONDITION_TYPE


# Constants
MAPPING_WEATHER = {"none": 0, "clear": 1, "night": 2, "rain": 3, "fog": 4}
MAPPING_FULL = {
    "none": 0,
    "car_clear": 1,
    "car_night": 2,
    "car_rain": 3,
    "car_fog": 4,
    "drone_clear": 5,
    "drone_night": 6,
    "drone_rain": 7,
    "drone_fog": 8,
}
REVERSE_MAPPING_WEATHER = {v: k for k, v in MAPPING_WEATHER.items()}
REVERSE_MAPPING_FULL = {v: k for k, v in MAPPING_FULL.items()}
FULL2WEATHER = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
}

NUM_CONDITIONS = {"none": 1, "weather": len(MAPPING_WEATHER), "full": len(MAPPING_FULL)}


class ConditionMap:
    def __init__(self, type: str):
        # Check that the type is correct
        if type not in CONDITION_TYPE:
            raise ValueError(f"Type must be one of {CONDITION_TYPE}")

        self.type = type

    def __call__(self, condition: list) -> list:
        out = [0 for _ in range(len(condition))]
        if self.type == "full":
            for i, elm in enumerate(condition):
                out[i] = MAPPING_FULL[elm]
        elif self.type == "weather":
            for i, elm in enumerate(condition):
                out[i] = MAPPING_WEATHER[elm.split("_")[1]]

        return out
