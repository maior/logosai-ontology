"""
🔄 Agent Sync Service
에이전트 마켓플레이스 ↔ Knowledge Graph ↔ Agent Registry 동기화 서비스

동기화 대상:
1. ACP Server (logosai/logosai/examples/agents/) - Source of Truth
2. Agent Metadata JSON (agents/config/agent_metadata.json)
3. Agent Registry (orchestrator/agent_registry.py)
4. Knowledge Graph (knowledge_graph_clean.py)

동기화 방식:
- 시작 시 전체 동기화 (full_sync)
- 실시간 이벤트 기반 동기화 (event-driven)
- 주기적 동기화 (periodic_sync)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from loguru import logger

# ACP Server 에이전트 디렉토리
# ontology/core/agent_sync_service.py 기준 경로 계산
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # /Users/maior/Development/skku/Logos
DEFAULT_AGENTS_DIR = _PROJECT_ROOT / "logosai" / "logosai" / "examples" / "agents"
DEFAULT_METADATA_FILE = _PROJECT_ROOT / "agents" / "config" / "agent_metadata.json"


class AgentSyncService:
    """
    🔄 에이전트 동기화 서비스

    ACP Server의 에이전트 정보를 Knowledge Graph와 Agent Registry에 동기화합니다.
    새 에이전트가 추가되면 자동으로 감지하여 시스템 전체에 반영합니다.
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
            agents_dir: ACP 에이전트 디렉토리 경로
            metadata_file: 에이전트 메타데이터 JSON 파일 경로
            knowledge_graph: KnowledgeGraphEngine 인스턴스
            agent_registry: AgentRegistry 인스턴스
        """
        self.agents_dir = agents_dir or DEFAULT_AGENTS_DIR
        self.metadata_file = metadata_file or DEFAULT_METADATA_FILE

        self._knowledge_graph = knowledge_graph
        self._agent_registry = agent_registry

        # 동기화 상태 추적
        self._synced_agents: Set[str] = set()
        self._last_sync: Optional[datetime] = None
        self._sync_in_progress = False

        # 파일 감시 상태
        self._file_hashes: Dict[str, str] = {}

        logger.info(f"🔄 에이전트 동기화 서비스 초기화 (agents_dir: {self.agents_dir})")

    @property
    def knowledge_graph(self):
        """지식그래프 지연 로딩"""
        if self._knowledge_graph is None:
            try:
                from ..engines.knowledge_graph_clean import KnowledgeGraphEngine
                self._knowledge_graph = KnowledgeGraphEngine(fast_mode=True)
                logger.info("📊 지식그래프 엔진 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 지식그래프 로드 실패: {e}")
        return self._knowledge_graph

    @property
    def agent_registry(self):
        """에이전트 레지스트리 지연 로딩"""
        if self._agent_registry is None:
            try:
                from ..orchestrator.agent_registry import get_registry
                self._agent_registry = get_registry()
                logger.info("📋 에이전트 레지스트리 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 에이전트 레지스트리 로드 실패: {e}")
        return self._agent_registry

    async def full_sync(self) -> Dict[str, Any]:
        """
        🔄 전체 동기화 실행

        1. ACP Server 에이전트 스캔
        2. 메타데이터 파일 로드
        3. Agent Registry 업데이트
        4. Knowledge Graph 업데이트

        Returns:
            동기화 결과 {added, updated, removed, errors}
        """
        if self._sync_in_progress:
            logger.warning("⚠️ 동기화가 이미 진행 중입니다")
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
            logger.info("🔄 전체 에이전트 동기화 시작...")

            # 1. ACP Server 에이전트 스캔
            acp_agents = await self._scan_acp_agents()
            logger.info(f"   📁 ACP 에이전트 스캔 완료: {len(acp_agents)}개")

            # 2. 메타데이터 파일 로드
            metadata_agents = await self._load_metadata_file()
            logger.info(f"   📄 메타데이터 파일 로드 완료: {len(metadata_agents)}개")

            # 3. 에이전트 정보 병합
            merged_agents = self._merge_agent_info(acp_agents, metadata_agents)
            result["total_agents"] = len(merged_agents)

            # 4. Agent Registry 업데이트
            registry_result = await self._sync_to_registry(merged_agents)
            result["added"].extend(registry_result.get("added", []))
            result["updated"].extend(registry_result.get("updated", []))

            # 5. Knowledge Graph 업데이트
            kg_result = await self._sync_to_knowledge_graph(merged_agents)
            if kg_result.get("errors"):
                result["errors"].extend(kg_result["errors"])

            # 6. 동기화 상태 업데이트
            self._synced_agents = set(merged_agents.keys())
            self._last_sync = datetime.now()

            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"✅ 전체 동기화 완료: "
                f"{len(result['added'])}개 추가, "
                f"{len(result['updated'])}개 업데이트, "
                f"{elapsed_ms:.0f}ms"
            )

        except Exception as e:
            logger.error(f"❌ 전체 동기화 실패: {e}")
            result["errors"].append(str(e))

        finally:
            self._sync_in_progress = False

        return result

    async def _scan_acp_agents(self) -> Dict[str, Dict[str, Any]]:
        """ACP Server 에이전트 디렉토리 스캔"""
        agents = {}

        if not self.agents_dir.exists():
            logger.warning(f"⚠️ 에이전트 디렉토리 없음: {self.agents_dir}")
            return agents

        # .py 파일 스캔 (테스트 파일 제외)
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
                logger.warning(f"⚠️ 에이전트 파일 파싱 실패 ({file_path.name}): {e}")

        return agents

    async def _parse_agent_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """에이전트 Python 파일에서 메타데이터 추출"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # 에이전트 ID 추출 (파일명 기반)
            agent_id = file_path.stem  # e.g., "internet_agent"

            # 클래스명 추출
            import re
            class_match = re.search(r'class\s+(\w+Agent)\s*[:\(]', content)
            class_name = class_match.group(1) if class_match else f"{agent_id.title()}Agent"

            # description 추출 (docstring 또는 description 변수)
            desc_match = re.search(r'description\s*[=:]\s*["\']([^"\']+)["\']', content)
            if not desc_match:
                # 클래스 docstring 시도
                docstring_match = re.search(r'class\s+\w+.*?:\s*"""([^"]+)"""', content, re.DOTALL)
                description = docstring_match.group(1).strip()[:200] if docstring_match else ""
            else:
                description = desc_match.group(1)[:200]

            # capabilities 추출
            caps_match = re.search(r'capabilities\s*[=:]\s*\[([^\]]+)\]', content)
            capabilities = []
            if caps_match:
                caps_str = caps_match.group(1)
                capabilities = [c.strip().strip('"\'') for c in caps_str.split(',') if c.strip()]

            # tags 추출
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
            logger.debug(f"에이전트 파일 파싱 오류 ({file_path}): {e}")
            return None

    async def _load_metadata_file(self) -> Dict[str, Dict[str, Any]]:
        """에이전트 메타데이터 JSON 파일 로드"""
        agents = {}

        if not self.metadata_file.exists():
            logger.debug(f"메타데이터 파일 없음: {self.metadata_file}")
            return agents

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for agent_name, agent_info in data.items():
                # agent_id 정규화 (예: "internet" → "internet_agent")
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
            logger.warning(f"⚠️ 메타데이터 파일 로드 실패: {e}")

        return agents

    def _merge_agent_info(
        self,
        acp_agents: Dict[str, Dict[str, Any]],
        metadata_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """여러 소스의 에이전트 정보 병합 (ACP 우선)"""
        merged = {}

        # ACP 에이전트 기준
        all_agent_ids = set(acp_agents.keys()) | set(metadata_agents.keys())

        for agent_id in all_agent_ids:
            acp_info = acp_agents.get(agent_id, {})
            meta_info = metadata_agents.get(agent_id, {})

            # ACP 정보 우선, 메타데이터로 보충
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
        """Agent Registry에 동기화"""
        result = {"added": [], "updated": []}

        if not self.agent_registry:
            return result

        try:
            from ..orchestrator.models import AgentSchema, AgentRegistryEntry

            for agent_id, info in agents.items():
                # 기존 에이전트 확인
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

                # 등록/업데이트
                self.agent_registry.register_agent(entry)

                if existing:
                    result["updated"].append(agent_id)
                else:
                    result["added"].append(agent_id)

        except Exception as e:
            logger.warning(f"⚠️ 레지스트리 동기화 실패: {e}")

        return result

    async def _sync_to_knowledge_graph(
        self,
        agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Knowledge Graph에 에이전트 정보 동기화"""
        result = {"synced": 0, "errors": []}

        if not self.knowledge_graph:
            return result

        try:
            for agent_id, info in agents.items():
                # 에이전트 노드 추가
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

                # 능력 노드 및 관계 추가
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

                # 태그 노드 및 관계 추가
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

            logger.info(f"📊 지식그래프 동기화 완료: {result['synced']}개 에이전트")

        except Exception as e:
            logger.warning(f"⚠️ 지식그래프 동기화 실패: {e}")
            result["errors"].append(str(e))

        return result

    async def sync_single_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """
        단일 에이전트 동기화 (실시간 추가/업데이트용)

        Args:
            agent_id: 에이전트 ID
            agent_info: 에이전트 정보

        Returns:
            성공 여부
        """
        try:
            logger.info(f"🔄 단일 에이전트 동기화: {agent_id}")

            # Registry 업데이트
            await self._sync_to_registry({agent_id: agent_info})

            # Knowledge Graph 업데이트
            await self._sync_to_knowledge_graph({agent_id: agent_info})

            self._synced_agents.add(agent_id)

            logger.info(f"✅ 에이전트 동기화 완료: {agent_id}")
            return True

        except Exception as e:
            logger.error(f"❌ 에이전트 동기화 실패 ({agent_id}): {e}")
            return False

    async def check_for_changes(self) -> Dict[str, List[str]]:
        """
        변경사항 확인 (주기적 동기화용)

        Returns:
            {"added": [...], "modified": [...], "removed": [...]}
        """
        changes = {"added": [], "modified": [], "removed": []}

        try:
            # 현재 ACP 에이전트 스캔
            current_agents = await self._scan_acp_agents()
            current_ids = set(current_agents.keys())

            # 추가된 에이전트
            changes["added"] = list(current_ids - self._synced_agents)

            # 제거된 에이전트
            changes["removed"] = list(self._synced_agents - current_ids)

            # 수정된 에이전트 (파일 해시 비교)
            for agent_id in current_ids & self._synced_agents:
                file_path = current_agents[agent_id].get("file_path")
                if file_path:
                    new_hash = self._get_file_hash(file_path)
                    old_hash = self._file_hashes.get(agent_id)
                    if old_hash and new_hash != old_hash:
                        changes["modified"].append(agent_id)
                    self._file_hashes[agent_id] = new_hash

        except Exception as e:
            logger.warning(f"⚠️ 변경사항 확인 실패: {e}")

        return changes

    def _get_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def get_sync_status(self) -> Dict[str, Any]:
        """동기화 상태 조회"""
        return {
            "synced_agents": list(self._synced_agents),
            "total_synced": len(self._synced_agents),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_in_progress": self._sync_in_progress,
            "agents_dir": str(self.agents_dir),
        }


# 싱글톤 인스턴스
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
