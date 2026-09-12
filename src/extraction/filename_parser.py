"""Parse raw video filenames into metadata.

Convention (docs/project_context.md):
    {exercise}_{posture_class}_p{participant_id}_{camera_angle}_take{NN}.mp4

Example: squat_correct_p1_left45_take01.mp4
"""
import re
from pathlib import Path

CAMERA_ANGLES = {"left45", "right45", "front"}

POSTURE_CLASSES_BY_EXERCISE = {
    "squat": {"correct", "kneeinward", "backbend"},
    "deadlift": {"correct", "backround", "armsspread"},
    "benchpress": {"correct", "flatback", "armsspread"},
}

_FILENAME_RE = re.compile(
    r"^(?P<exercise>[a-z]+)_(?P<posture_class>[a-z]+)_p(?P<participant_id>\d+)_"
    r"(?P<camera_angle>left45|right45|front)_take(?P<take>\d+)\.mp4$",
    re.IGNORECASE,
)


class FilenameParseError(ValueError):
    pass


def parse_video_filename(path):
    """Parse a raw video path/filename into a metadata dict.

    Raises FilenameParseError with a human-readable reason if the filename
    doesn't match the naming convention, or uses an unknown exercise/posture.
    """
    name = Path(path).name
    match = _FILENAME_RE.match(name)
    if not match:
        raise FilenameParseError(
            f"'{name}' tidak cocok pola {{exercise}}_{{posture_class}}_p{{id}}_"
            f"{{camera_angle}}_take{{NN}}.mp4"
        )

    exercise = match.group("exercise").lower()
    posture_class = match.group("posture_class").lower()
    camera_angle = match.group("camera_angle").lower()

    if exercise not in POSTURE_CLASSES_BY_EXERCISE:
        raise FilenameParseError(
            f"'{name}': exercise '{exercise}' tidak dikenal "
            f"(harus salah satu dari {sorted(POSTURE_CLASSES_BY_EXERCISE)})"
        )
    valid_postures = POSTURE_CLASSES_BY_EXERCISE[exercise]
    if posture_class not in valid_postures:
        raise FilenameParseError(
            f"'{name}': posture_class '{posture_class}' tidak dikenal untuk "
            f"exercise '{exercise}' (harus salah satu dari {sorted(valid_postures)})"
        )

    return {
        "exercise": exercise,
        "posture_class": posture_class,
        "participant_id": int(match.group("participant_id")),
        "camera_angle": camera_angle,
        "take": int(match.group("take")),
        "source_video": name,
    }
