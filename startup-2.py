#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio

import fire

from metagpt.roles import (
    ProjectManager,

)

from custom.roles import (
    ProductManager,
    Architect,
    Engineer,
    QaEngineer,
    Researcher
)

from custom.artalaxies_llc import ArtalaxiesLLC


async def startup(
    idea: str,
    n_round: int = 5,
    implement: bool = True,
):
    """Run a startup. Be a boss."""
    company = ArtalaxiesLLC()
    company.hire(
        [
            Researcher(),
            ProductManager(),
            Architect(),
            ProjectManager(),
            Engineer(n_borg=5, use_code_review=True),
            QaEngineer()
        ]
    )

    company.start_project(idea)
    await company.run(n_round=n_round)


def main(
    idea: str,
    n_round: int = 5,
    implement: bool = True,
):
    """
    We are a software startup comprised of AI. By investing in us,
    you are empowering a future filled with limitless possibilities.
    :param idea: Your innovative idea, such as "Creating a snake game."
    :param investment: As an investor, you have the opportunity to contribute
    a certain dollar amount to this AI company.
    :param n_round:
    :param code_review: Whether to use code review.
    :return:
    """
    asyncio.run(startup(idea, n_round, implement))


if __name__ == "__main__":
    fire.Fire(main)
