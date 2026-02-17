"""
🔄 Agent Sync Service
Synchronization service for Agent Marketplace ↔ Knowledge Graph ↔ Agent Registry

Sync targets:
1. ACP Server (logosai/logosai/examples/agents/) - Source of Truth
2. Agent Metadata JSON (agents/config/agent_metadata.json)
3. Agent Registry (orchestrator/agent_registry.py)
4. Knowledge Graph (knowledge_graph_clean.py)

Sync methods:
- Full sync on startup (full_sync)
- Real-time event-driven sync (event-driven)
- Periodic sync (periodic_sync)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from loguru import logger

# ACP Server agent directory
# Path calculated relative to ontology/core/agent_sync_service.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # /Users/maior/Development/skku/Logos
DEFAULT_AGENTS_DIR = _PROJECT_ROOT / "acp_server" / "agents"
DEFAULT_METADATA_FILE = _PROJECT_ROOT / "agents" / "config" / "agent_metadata.json"


class AgentSyncService:
    """
    🔄 Agent Synchronization Service

    Synchronizes agent information from ACP Server to Knowledge Graph and Agent Registry.
    Automatically detects newly added agents and reflects them across the system.
    """

    def __init__(
        self,
        agents_dir: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
        knowledge_graph=None,
        agent_registry=None,
    ):
        """
        Args:
            agents_dir: ACP agent directory path
            metadata_file: Agent metadata JSON file path
            knowledge_graph: KnowledgeGraphEngine instance
            agent_registry: AgentRegistry instance
        """
        self.agents_dir = agents_dir or DEFAULT_AGENTS_DIR
        self.metadata_file = metadata_file or DEFAULT_METADATA_FILE

        self._knowledge_graph = knowledge_graph
        self._agent_registry = agent_registry

        # Sync state tracking
        self._synced_agents: Set[str] = set()
        self._last_sync: Optional[datetime] = None
        self._sync_in_progress = False

        # File watching state
        self._file_hashes: Dict[str, str] = {}

        logger.info(f"🔄 Agent sync service initialized (agents_dir: {self.agents_dir})")

    @property
    def knowledge_graph(self):
        """Lazy-load knowledge graph"""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
                self._knowledge_graph = KnowledgeGraphEngine(fast_mode=True)
                logger.info("📊 Knowledge graph engine loaded")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge graph load failed: {e}")
        return self._knowledge_graph

    @property
    def agent_registry(self):
        """Lazy-load agent registry"""
        if self._agent_registry is None:
            try:
                from ..orchestrator.agent_registry import get_registry
                self._agent_registry = get_registry()
                logger.info("📋 Agent registry loaded")
            except Exception as e:
                logger.warning(f"⚠️ Agent registry load failed: {e}")
        return self._agent_registry

    async def full_sync(self) -> Dict[str, Any]:
        """
        🔄 Run full synchronization

        1. Scan ACP Server agents
        2. Load metadata file
        3. Update Agent Registry
        4. Knowledge Graph 업데이트

        Returns:
            Sync result {added, updated, removed, errors}
        """
        if self._sync_in_progress:
            logger.warning("⚠️ Sync is already in progress")
            return {"status": "already_in_progress"}

        self._sync_in_progress = True
        start_time = datetime.now()

        result = {
            "added": [],
            "updated": [],
            "removed": [],
            "errors": [],
            "total_agents": 0,
        }

        try:
            logger.info("🔄 Starting full agent synchronization...")

            # 1. Scan ACP Server agents
            acp_agents = await self._scan_acp_agents()
            logger.info(f"   📁 ACP agent scan complete: {len(acp_agents)} agents")

            # 2. Load metadata file
            metadata_agents = await self._load_metadata_file()
            logger.info(f"   📄 Metadata file loaded: {len(metadata_agents)} agents")

            # 3. Merge agent information
            merged_agents = self._merge_agent_info(acp_agents, metadata_agents)
            result["total_agents"] = len(merged_agents)

            # 4. Update Agent Registry
            registry_result = await self._sync_to_registry(merged_agents)
            result["added"].extend(registry_result.get("added", []))
            result["updated"].extend(registry_result.get("updated", []))

            # 5. Update Knowledge Graph
            kg_result = await self._sync_to_knowledge_graph(merged_agents)
            if kg_result.get("errors"):
                result["errors"].extend(kg_result["errors"])

            # 6. Update sync state
            self._synced_agents = set(merged_agents.keys())
            self._last_sync = datetime.now()

            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"✅ Full sync complete: "
                f"{len(result['added'])} added, "
                f"{len(result['updated'])} updated, "
                f"{elapsed_ms:.0f}ms"
            )

        except Exception as e:
            logger.error(f"❌ Full sync failed: {e}")
            result["errors"].append(str(e))

        finally:
            self._sync_in_progress = False

        return result

    async def _scan_acp_agents(self) -> Dict[str, Dict[str, Any]]:
        """Scan ACP Server agent directory"""
        agents = {}

        if not self.agents_dir.exists():
            logger.warning(f"⚠️ Agent directory not found: {self.agents_dir}")
            return agents

        # Scan .py files (exclude test files)
        for file_path in self.agents_dir.glob("*_agent.py"):
            if file_path.name.startswith("test_") or file_path.name.startswith("_"):
                continue

            try:
                agent_info = await self._parse_agent_file(file_path)
                if agent_info:
                    agent_id = agent_info.get("agent_id")
                    if agent_id:
                        agents[agent_id] = agent_info

            except Exception as e:
                logger.warning(f"⚠️ Agent file parsing failed ({file_path.name}): {e}")

        return agents

    async def _parse_agent_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from agent Python file"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # Extract agent ID (based on filename)
            agent_id = file_path.stem  # e.g., "internet_agent"

            # Extract class name
            import re
            class_match = re.search(r'class\s+(\w+Agent)\s*[:\(]', content)
            class_name = class_match.group(1) if class_match else f"{agent_id.title()}Agent"

            # Extract description (from docstring or description variable)
            desc_match = re.search(r'description\s*[=:]\s*["\']([^"\']+)["\']', content)
            if not desc_match:
                # Try class docstring
                docstring_match = re.search(r'class\s+\w+.*?:\s*"""([^"]+)"""', content, re.DOTALL)
                description = docstring_match.group(1).strip()[:200] if docstring_match else ""
            else:
                description = desc_match.group(1)[:200]

            # Extract capabilities
            caps_match = re.search(r'capabilities\s*[=:]\s*\[([^\]]+)\]', content)
            capabilities = []
            if caps_match:
                caps_str = caps_match.group(1)
                capabilities = [c.strip().strip('"\'') for c in caps_str.split(',') if c.strip()]

            # Extract tags
            tags_match = re.search(r'tags\s*[=:]\s*\[([^\]]+)\]', content)
            tags = []
            if tags_match:
                tags_str = tags_match.group(1)
                tags = [t.strip().strip('"\'') for t in tags_str.split(',') if t.strip()]

            return {
                "agent_id": agent_id,
                "class_name": class_name,
                "description": description,
                "capabilities": capabilities,
                "tags": tags,
                "file_path": str(file_path),
                "source": "acp_server",
            }

        except Exception as e:
            logger.debug(f"Agent file parsing error ({file_path}): {e}")
            return None

    async def _load_metadata_file(self) -> Dict[str, Dict[str, Any]]:
        """Load agent metadata JSON file"""
        agents = {}

        if not self.metadata_file.exists():
            logger.debug(f"Metadata file not found: {self.metadata_file}")
            return agents

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for agent_name, agent_info in data.items():
                # Normalize agent_id (e.g. "internet" → "internet_agent")
                agent_id = agent_name if agent_name.endswith("_agent") else f"{agent_name}_agent"

                agents[agent_id] = {
                    "agent_id": agent_id,
                    "name": agent_info.get("name", agent_name),
                    "class_name": agent_info.get("class_name", ""),
                    "description": agent_info.get("description", ""),
                    "capabilities": agent_info.get("capabilities", []),
                    "tags": agent_info.get("tags", []),
                    "version": agent_info.get("version", "1.0.0"),
                    "source": "metadata_file",
                }

        except Exception as e:
            logger.warning(f"⚠️ Metadata file load failed: {e}")

        return agents

    def _merge_agent_info(
        self,
        acp_agents: Dict[str, Dict[str, Any]],
        metadata_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Merge agent information from multiple sources (ACP takes priority)"""
        merged = {}

        # Based on ACP agents
        all_agent_ids = set(acp_agents.keys()) | set(metadata_agents.keys())

        for agent_id in all_agent_ids:
            acp_info = acp_agents.get(agent_id, {})
            meta_info = metadata_agents.get(agent_id, {})

            # ACP info takes priority, supplemented by metadata
            merged[agent_id] = {
                "agent_id": agent_id,
                "name": acp_info.get("name") or meta_info.get("name") or agent_id,
                "class_name": acp_info.get("class_name") or meta_info.get("class_name", ""),
                "description": acp_info.get("description") or meta_info.get("description", ""),
                "capabilities": acp_info.get("capabilities") or meta_info.get("capabilities", []),
                "tags": acp_info.get("tags") or meta_info.get("tags", []),
                "version": meta_info.get("version", "1.0.0"),
                "file_path": acp_info.get("file_path"),
                "sources": [s for s in ["acp_server" if acp_info else None, "metadata_file" if meta_info else None] if s],
                "synced_at": datetime.now().isoformat(),
            }

        return merged

    async def _sync_to_registry(
        self,
        agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Sync to Agent Registry"""
        result = {"added": [], "updated": []}

        if not self.agent_registry:
            return result

        try:
            from ..orchestrator.models import AgentSchema, AgentRegistryEntry

            for agent_id, info in agents.items():
                # Check existing agent
                existing = self.agent_registry.get_agent_safe(agent_id)

                # AgentRegistryEntry 생성
                entry = AgentRegistryEntry(
                    agent_id=agent_id,
                    name=info.get("name", agent_id),
                    description=info.get("description", ""),
                    capabilities=info.get("capabilities", []),
                    tags=info.get("tags", []),
                    schema=AgentSchema(
                        input_type="query",
                        output_type="text",
                    ),
                    display_name=info.get("name", agent_id),
                    display_name_ko=info.get("name", agent_id),
                    icon="🤖",
                    priority=0,
                )

                # Register/update
                self.agent_registry.register_agent(entry)

                if existing:
                    result["updated"].append(agent_id)
                else:
                    result["added"].append(agent_id)

        except Exception as e:
            logger.warning(f"⚠️ Registry sync failed: {e}")

        return result

    async def _sync_to_knowledge_graph(
        self,
        agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sync agent info to Knowledge Graph"""
        result = {"synced": 0, "errors": []}

        if not self.knowledge_graph:
            return result

        try:
            for agent_id, info in agents.items():
                # Add agent node
                await self.knowledge_graph.add_concept(
                    agent_id,
                    "agent",
                    {
                        "name": info.get("name", agent_id),
                        "description": info.get("description", ""),
                        "capabilities": info.get("capabilities", []),
                        "tags": info.get("tags", []),
                        "version": info.get("version", "1.0.0"),
                        "synced_at": info.get("synced_at"),
                        "is_available": True,
                    }
                )

                # Add capability nodes and relationships
                for capability in info.get("capabilities", []):
                    cap_id = f"capability_{capability}"
                    await self.knowledge_graph.add_concept(
                        cap_id,
                        "capability",
                        {"name": capability}
                    )
                    await self.knowledge_graph.add_relationship(
                        agent_id, cap_id, "has_capability"
                    )

                # Add tag nodes and relationships
                for tag in info.get("tags", []):
                    tag_id = f"tag_{tag}"
                    await self.knowledge_graph.add_concept(
                        tag_id,
                        "tag",
                        {"name": tag}
                    )
                    await self.knowledge_graph.add_relationship(
                        agent_id, tag_id, "has_tag"
                    )

                result["synced"] += 1

            logger.info(f"📊 Knowledge graph sync complete: {result['synced']} agents")

        except Exception as e:
            logger.warning(f"⚠️ Knowledge graph sync failed: {e}")
            result["errors"].append(str(e))

        return result

    async def sync_single_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """
        Sync a single agent (for real-time add/update)

        Args:
            agent_id: Agent ID
            agent_info: Agent information

        Returns:
            Success status
        """
        try:
            logger.info(f"🔄 Syncing single agent: {agent_id}")

            # Update registry
            await self._sync_to_registry({agent_id: agent_info})

            # Update Knowledge Graph
            await self._sync_to_knowledge_graph({agent_id: agent_info})

            self._synced_agents.add(agent_id)

            logger.info(f"✅ Agent sync complete: {agent_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Agent sync failed ({agent_id}): {e}")
            return False

    async def check_for_changes(self) -> Dict[str, List[str]]:
        """
        Check for changes (for periodic sync)

        Returns:
            {"added": [...], "modified": [...], "removed": [...]}
        """
        changes = {"added": [], "modified": [], "removed": []}

        try:
            # Scan current ACP agents
            current_agents = await self._scan_acp_agents()
            current_ids = set(current_agents.keys())

            # Added agents
            changes["added"] = list(current_ids - self._synced_agents)

            # Removed agents
            changes["removed"] = list(self._synced_agents - current_ids)

            # Modified agents (file hash comparison)
            for agent_id in current_ids & self._synced_agents:
                file_path = current_agents[agent_id].get("file_path")
                if file_path:
                    new_hash = self._get_file_hash(file_path)
                    old_hash = self._file_hashes.get(agent_id)
                    if old_hash and new_hash != old_hash:
                        changes["modified"].append(agent_id)
                    self._file_hashes[agent_id] = new_hash

        except Exception as e:
            logger.warning(f"⚠️ Change detection failed: {e}")

        return changes

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return {
            "synced_agents": list(self._synced_agents),
            "total_synced": len(self._synced_agents),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_in_progress": self._sync_in_progress,
            "agents_dir": str(self.agents_dir),
        }


# Singleton instance
_sync_service_instance: Optional[AgentSyncService] = None


def get_sync_service() -> AgentSyncService:
    """동기화 서비스 싱글톤 인스턴스 반환"""
    global _sync_service_instance
    if _sync_service_instance is None:
        _sync_service_instance = AgentSyncService()
    return _sync_service_instance


async def initialize_agent_sync():
    """
    시스템 시작 시 에이전트 동기화 초기화

    사용법:
        # 서버 시작 시 호출
        await initialize_agent_sync()
    """
    service = get_sync_service()
    result = await service.full_sync()
    return result


class AgentFileWatcher:
    """
    👁️ 에이전트 파일 감시자

    ACP 에이전트 디렉토리의 변경사항을 감지하여 자동으로 동기화합니다.

    사용법:
        watcher = AgentFileWatcher()
        await watcher.start()  # 백그라운드 감시 시작
        await watcher.stop()   # 감시 중지
    """

    def __init__(
        self,
        sync_service: Optional[AgentSyncService] = None,
        check_interval: float = 5.0,  # 5초마다 체크
    ):
        self._sync_service = sync_service
        self._check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def sync_service(self) -> AgentSyncService:
        if self._sync_service is None:
            self._sync_service = get_sync_service()
        return self._sync_service

    async def start(self):
        """파일 감시 시작"""
        if self._running:
            logger.warning("👁️ 파일 감시가 이미 실행 중입니다")
            return

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"👁️ 에이전트 파일 감시 시작 (간격: {self._check_interval}초)")

    async def stop(self):
        """파일 감시 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("👁️ 에이전트 파일 감시 중지됨")

    async def _watch_loop(self):
        """감시 루프"""
        while self._running:
            try:
                # 변경사항 확인
                changes = await self.sync_service.check_for_changes()

                if changes.get("added") or changes.get("modified") or changes.get("removed"):
                    logger.info(
                        f"👁️ 에이전트 변경 감지: "
                        f"+{len(changes.get('added', []))} "
                        f"~{len(changes.get('modified', []))} "
                        f"-{len(changes.get('removed', []))}"
                    )

                    # 변경된 에이전트만 동기화
                    for agent_id in changes.get("added", []) + changes.get("modified", []):
                        acp_agents = await self.sync_service._scan_acp_agents()
                        if agent_id in acp_agents:
                            await self.sync_service.sync_single_agent(agent_id, acp_agents[agent_id])

                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ 파일 감시 오류: {e}")
                await asyncio.sleep(self._check_interval)


# 파일 감시자 싱글톤
_file_watcher_instance: Optional[AgentFileWatcher] = None


def get_file_watcher() -> AgentFileWatcher:
    """파일 감시자 싱글톤 인스턴스 반환"""
    global _file_watcher_instance
    if _file_watcher_instance is None:
        _file_watcher_instance = AgentFileWatcher()
    return _file_watcher_instance


async def start_agent_file_watching():
    """
    에이전트 파일 감시 시작 (서버 시작 시 호출)

    사용법:
        # 서버 시작 시 호출
        await start_agent_file_watching()
    """
    watcher = get_file_watcher()
    await watcher.start()
    return watcher


logger.info("🔄 에이전트 동기화 서비스 모듈 로드 완료")
