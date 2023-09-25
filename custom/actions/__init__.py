#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

from custom.actions.research import CollectLinks, WebBrowseAndSummarize, ConductResearch
from custom.actions.run_code import RunCode

class ActionType(Enum):
    """All types of Actions, used for indexing."""

    RUN_CODE = RunCode
    COLLECT_LINKS = CollectLinks
    WEB_BROWSE_AND_SUMMARIZE = WebBrowseAndSummarize
    CONDUCT_RESEARCH = ConductResearch


__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]
