from dataclasses import dataclass
from typing import Optional


@dataclass
class SubtitleArea:
    ymin: int = 0
    ymax: int = 0
    xmin: int = 0
    xmax: int = 0
    ab_section: Optional[range] = None

    def __post_init__(self):
        self.normalized()

    def normalized(self):
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin

    def is_empty(self):
        return self.xmin == 0 and self.xmax == 0 and self.ymin == 0 and self.ymax == 0

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def to_tuple(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)
