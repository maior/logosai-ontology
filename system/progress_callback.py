"""
🔄 Progress Callback
Progress Callback Module

Tracks and notifies progress of the ontology system
"""

import time
from typing import Dict, List, Any
from loguru import logger

from ..core.models import AgentExecutionResult
from ..core.interfaces import ProgressCallback


class SimpleProgressCallback(ProgressCallback):
    """Simple progress callback"""
    
    def __init__(self):
        self.messages = []
        self.current_progress = 0.0
        self.completed_steps = []
        self.errors = []
    
    async def on_progress(self, message: str, progress: float, metadata: Dict[str, Any] = None):
        """Update progress"""
        self.messages.append({
            "message": message,
            "progress": progress,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self.current_progress = progress
        logger.info(f"Progress ({progress:.1%}): {message}")
    
    async def on_step_complete(self, step_id: str, result: AgentExecutionResult):
        """Step completion notification"""
        self.completed_steps.append({
            "step_id": step_id,
            "agent_id": result.agent_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
        logger.info(f"Step completed: {step_id} ({result.agent_id}) - {'✅' if result.success else '❌'}")
    
    async def on_error(self, error_message: str, error_details: Dict[str, Any] = None):
        """Error notification"""
        self.errors.append({
            "message": error_message,
            "details": error_details or {},
            "timestamp": time.time()
        })
        logger.error(f"Error: {error_message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Progress summary"""
        return {
            "current_progress": self.current_progress,
            "total_messages": len(self.messages),
            "completed_steps": len(self.completed_steps),
            "errors": len(self.errors),
            "success_rate": (
                sum(1 for step in self.completed_steps if step["success"]) / 
                len(self.completed_steps) if self.completed_steps else 0
            )
        } 