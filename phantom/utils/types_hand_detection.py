from enum import Enum
import numpy as np
from typing import List


# Enum to represent the side of the hand
class HandSide(Enum):
    LEFT = 0
    RIGHT = 1


# Enum to represent the state of the hand
class HandState(Enum):
    NO_CONTACT = 0
    SELF_CONTACT = 1
    ANOTHER_PERSON = 2
    PORTABLE_OBJECT = 3
    STATIONARY_OBJECT = 4


# A BBox class matching the Protobuf definition
class BBox:
    def __init__(self, left: float, top: float, right: float, bottom: float):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def center(self):
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


# A FloatVector class matching the Protobuf definition
class FloatVector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def scale(self, width_factor: float = 1, height_factor: float = 1):
        self.x *= width_factor
        self.y *= height_factor


# HandDetection class matching the Protobuf definition
class HandDetection:
    def __init__(self, bbox: BBox, score: float, state: HandState, object_offset: FloatVector, side: HandSide):
        self.bbox = bbox
        self.score = score
        self.state = state
        self.object_offset = object_offset
        self.side = side

    @staticmethod
    def from_detection(detection: List[float]) -> "HandDetection":
        """
        Simplified method to create a HandDetection object from a detection list
        :param detection: List of detection attributes (bbox, score, state, offset)
        :return: HandDetection object
        """
        assert len(detection) == 10  # Length of the detection list should be 10

        # Extract values from the detection list
        bbox = BBox(left=detection[0], top=detection[1], right=detection[2], bottom=detection[3])
        score = detection[4]
        state = HandState(int(detection[5]))  # Convert state to enum
        # Note: Protobuf has object_offset with x,y. The old format had magnitude at index 6.
        # Assuming new format where indices 6,7 are x,y for the offset.
        offset = FloatVector(x=detection[7], y=detection[8])
        side = HandSide(int(detection[9]))  # Left or Right hand

        return HandDetection(bbox, score, state, offset, side)

    def scale(self, width_factor: float = 1, height_factor: float = 1):
        self.bbox.left = round(self.bbox.left * width_factor)
        self.bbox.top = round(self.bbox.top * height_factor)
        self.bbox.right = round(self.bbox.right * width_factor)
        self.bbox.bottom = round(self.bbox.bottom * height_factor)
        self.object_offset.scale(width_factor, height_factor)