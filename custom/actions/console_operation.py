from __future__ import annotations

import asyncio
import json
from typing import Callable

from pydantic import parse_obj_as
import sys

from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools.web_browser_engine import WebBrowserEngine, WebBrowserEngineType
from metagpt.utils.text import generate_prompt_chunk, reduce_message_length
from langchain.vectorstores.base import VectorStore
import dirtyjson
from paramiko import SSHClient, AutoAddPolicy, Channel




OPERATION_INTENSION = "You are an AI console operator, and your work load is: {topic}"

IDENDTIFY_SYSTEM_PROMPT = ""




class OperateConsole(Action):

    def __init__(
            self,
            name: str = "",
            *args,
            **kwargs):
        super().__init__(name, args, kwargs)
        self.desc = "Apply operation on a console."

    async def run(
            self,
            topic: str
            ):
        """Run the action to operate a console.

        Args:
            topic: The research topic.
            
        Returns:
            A document containing the console logs.
        """
        hostname = CONFIG.ssh_hostname
        username = CONFIG.ssh_username
        port = CONFIG.ssh_port
        password = CONFIG.ssh_password
        key_filename = CONFIG.ssh_key_filename
        self.console_channel = self.connect_shell(
            hostname,
            port,
            username,
            password,
            key_filename
        )

        system_text = system_text if system_text else OPERATION_INTENSION.format(topic=topic)
   
    
        def connect_shell(
                hostname :str, 
                port : int, 
                username: str | None = None, 
                password: str | None = None,
                key_filename: str | None = None,
        ) -> Channel: 
            
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(
                        hostname = hostname,
                        port= port, 
                        username=username, 
                        # pkey= PKey.from_path("/home/" + username + "/id_rsa"),
                        password=password,
                        key_filename= key_filename
                        )

            channel = ssh.invoke_shell()
            # writer = threading.Thread(target=chan_recv, args=(channel,placeholder,), daemon=True) 
            # add_script_run_ctx(writer)
            # writer.start()
            channel.settimeout(3)
            return channel
