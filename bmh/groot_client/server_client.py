# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from Isaac-GR00T `gr00t/policy/server_client.py`. PolicyServer is omitted
# (the BMH-101 only acts as a client). Imports are rewritten to the sibling vendored
# modules so this file pulls in no further GR00T dependencies.

from typing import Any

import msgpack_numpy as mnp
import zmq

from .json_utils import to_json_serializable
from .policy_base import BasePolicy
from .types import ModalityConfig


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return mnp.packb(data, default=MsgSerializer._encode_custom)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return mnp.unpackb(data, object_hook=MsgSerializer._decode_custom, raw=False)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig__": True, "as_json": to_json_serializable(obj)}
        return mnp.encode(obj)

    @staticmethod
    def _decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig__" in obj or b"__ModalityConfig__" in obj:
            key = "as_json" if "as_json" in obj else (b"as_json" if b"as_json" in obj else None)
            if key is not None:
                return ModalityConfig(**obj[key])
        return mnp.decode(obj)


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # Timeout — REQ socket is now in an invalid state (waiting for a
            # reply that will never arrive). Recreate it so the next call can
            # send again, then re-raise so the caller knows this request failed.
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. "
            "Use `strict=False` (the default for the BMH client) or override in a subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. "
            "Use `strict=False` (the default for the BMH client) or override in a subclass."
        )
