"""Agentic vision workflows with tool use and iterative reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping


ToolFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class VisionTool:
    """Callable tool available to a vision agent."""

    name: str
    description: str
    fn: ToolFn


@dataclass
class WorkflowResult:
    """Agent workflow trace and outputs."""

    goal: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


class VisionAgent:
    """Small deterministic agent runtime for visual inspection pipelines."""

    def __init__(self, tools: Iterable[VisionTool]) -> None:
        self.tools: Dict[str, VisionTool] = {tool.name: tool for tool in tools}

    def run_plan(self, goal: str, plan: Iterable[Mapping[str, Any]], context: Mapping[str, Any]) -> WorkflowResult:
        state: dict[str, Any] = dict(context)
        result = WorkflowResult(goal=goal)
        for step in plan:
            tool_name = str(step["tool"])
            args = {**state, **dict(step.get("args", {}))}
            output = self.tools[tool_name].fn(args)
            state.update(output)
            result.steps.append({"tool": tool_name, "args": step.get("args", {}), "output_keys": list(output.keys())})
        result.artifacts = state
        return result

    def detect_crop_redetect_plan(self) -> list[dict[str, Any]]:
        """Default autonomous inspection loop."""

        return [
            {"tool": "detect", "args": {}},
            {"tool": "crop", "args": {"source_key": "detections"}},
            {"tool": "detect", "args": {"source_key": "crops", "refine": True}},
            {"tool": "summarize", "args": {}},
        ]

    def ocr_understand_act_plan(self) -> list[dict[str, Any]]:
        """Default OCR to document understanding to action loop."""

        return [
            {"tool": "ocr", "args": {}},
            {"tool": "document_understand", "args": {}},
            {"tool": "policy", "args": {}},
            {"tool": "act", "args": {}},
        ]


def local_command_tool(name: str, command_builder: Callable[[Mapping[str, Any]], list[str]]) -> VisionTool:
    """Create a local shell execution tool for controlled orchestration."""

    def run(args: Mapping[str, Any]) -> Mapping[str, Any]:
        import subprocess

        completed = subprocess.run(command_builder(args), check=True, capture_output=True, text=True)
        return {f"{name}_stdout": completed.stdout, f"{name}_stderr": completed.stderr}

    return VisionTool(name, f"Run local command for {name}", run)
