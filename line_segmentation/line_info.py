from __future__ import annotations

from typing import Optional

from dataclasses import dataclass


@dataclass
class LineInfo:
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    next: Optional['LineInfo']


@dataclass
class LineInfo_Compleat:
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    line_binary: list
    next: Optional['LineInfo_Compleat']
