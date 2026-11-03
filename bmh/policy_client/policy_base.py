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

# Vendored from Isaac-GR00T `gr00t/policy/policy.py` (BasePolicy only — PolicyWrapper omitted).
# This file is the minimal abstract base PolicyClient inherits from; no further GR00T imports.

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base class for robotic control policies."""

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        pass
