#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : test_run_code.py
"""
import pytest

from custom.actions.run_command import RunCommand


@pytest.mark.asyncio
async def test_run_console():
    result = await RunCommand.run_interactive_console(".",command= "pwd")
    print('')
    print(result)
    result = await RunCommand.run_interactive_console(".",command= "ls")
    print(result)
    
    

