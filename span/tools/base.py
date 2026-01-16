from abc import ABC, abstractmethod
from typing import Any

from span.models.tools import ToolResult


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        pass

    def to_anthropic_tool(self) -> dict[str, Any]:
        properties = {}
        required = []

        for key, value in self.parameters.items():
            param_def = {k: v for k, v in value.items() if k != "required"}
            properties[key] = param_def
            if value.get("required", False):
                required.append(key)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }
