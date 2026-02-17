#!/usr/bin/env python3
"""
🧠 Enhanced Query Decomposer

A system that decomposes complex user queries into individual tasks
and generates specific, processed messages to be delivered to each agent.
"""

import os
import re
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from ..core.models import SemanticQuery, AgentType, ExecutionStrategy
from ..core.llm_manager import get_ontology_llm_manager, OntologyLLMType

DEFAULT_ACP_ENDPOINT = os.getenv("ACP_SERVER_URL", "http://localhost:8888") + "/jsonrpc"


@dataclass
class TaskComponent:
    """Individual task component"""
    task_id: str
    task_type: str  # "weather", "currency", "calculation", "analysis", etc.
    original_text: str  # Corresponding portion of the original query
    processed_message: str  # Processed message to deliver to the agent
    target_agent: str  # Agent ID responsible for this task
    priority: int = 1  # Execution priority (lower = executes earlier)
    dependencies: List[str] = field(default_factory=list)  # Dependencies (tasks that must complete first)
    expected_output: str = ""  # Expected output format
    confidence: float = 0.8  # Analysis confidence score

    # Compatibility property
    @property
    def agent_id(self) -> str:
        """Agent ID (compatibility property)"""
        return self.target_agent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "original_text": self.original_text,
            "processed_message": self.processed_message,
            "target_agent": self.target_agent,
            "agent_id": self.agent_id,  # compatibility
            "priority": self.priority,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "confidence": self.confidence
        }


@dataclass
class QueryDecompositionResult:
    """Query decomposition result"""
    original_query: str
    semantic_query: SemanticQuery
    task_components: List[TaskComponent]
    execution_strategy: ExecutionStrategy
    execution_plan: List[List[str]]  # Step-by-step execution plan (parallel groups)
    estimated_time: float
    overall_confidence: float
    reasoning: str

    # Compatibility properties
    @property
    def success(self) -> bool:
        """Whether decomposition succeeded (compatibility property)"""
        return len(self.task_components) > 0

    @property
    def tasks(self) -> List[TaskComponent]:
        """Task list (compatibility property)"""
        return self.task_components

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "semantic_query": self.semantic_query.to_dict(),
            "task_components": [task.to_dict() for task in self.task_components],
            "tasks": [task.to_dict() for task in self.task_components],  # compatibility
            "execution_strategy": self.execution_strategy.value,
            "execution_plan": self.execution_plan,
            "estimated_time": self.estimated_time,
            "overall_confidence": self.overall_confidence,
            "reasoning": self.reasoning,
            "success": self.success  # compatibility
        }


class EnhancedQueryDecomposer:
    """🧠 Advanced query decomposer"""

    def __init__(self):
        """Initialize Enhanced Query Decomposer"""
        # Agent capability mapping
        self.agent_capabilities = {
            "weather_agent": {
                "keywords": ["날씨", "기온", "온도", "비", "눈", "바람", "습도", "미세먼지", "weather", "temperature", "rain", "snow"],
                "task_types": ["weather_inquiry"],
                "output_format": "Weather information",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "finance_agent": {
                "keywords": ["환율", "주가", "달러", "유로", "엔", "투자", "금융", "경제", "exchange", "rate", "dollar", "stock", "finance"],
                "task_types": ["financial_data", "currency_exchange"],
                "output_format": "Financial data",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "calculate_agent": {
                "keywords": ["계산", "수학", "더하기", "빼기", "곱하기", "나누기", "큰수", "작은수", "평균", "합계", "calculate", "math", "plus", "minus"],
                "task_types": ["calculation", "math"],
                "output_format": "Calculation result",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "internet_agent": {
                "keywords": ["검색", "찾아", "정보", "알려", "조사", "최신", "뉴스", "웹", "인터넷", "search", "find", "information", "news", "web"],
                "task_types": ["search", "information_retrieval"],
                "output_format": "Search results",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "chart_agent": {
                "keywords": ["차트", "그래프", "시각화", "표", "그림", "도표", "chart", "graph", "visualization", "plot"],
                "task_types": ["visualization", "chart_generation"],
                "output_format": "Visualization data",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "analysis_agent": {
                "keywords": ["분석", "검토", "평가", "비교", "조사", "연구", "analysis", "review", "evaluate", "compare", "research"],
                "task_types": ["analysis", "evaluation"],
                "output_format": "Analysis result",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "translate_agent": {
                "keywords": ["번역", "영어", "한국어", "일본어", "중국어", "언어", "translate", "english", "korean", "japanese", "chinese"],
                "task_types": ["translation"],
                "output_format": "Translation result",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "image_agent": {
                "keywords": ["이미지", "그림", "사진", "이미지생성", "그려", "만들어", "image", "picture", "photo", "generate", "create"],
                "task_types": ["image_generation"],
                "output_format": "Image file",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "audio_agent": {
                "keywords": ["음성", "소리", "오디오", "음악", "녹음", "재생", "audio", "sound", "music", "record", "play"],
                "task_types": ["audio_generation", "audio_processing"],
                "output_format": "Audio file",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "file_agent": {
                "keywords": ["파일", "저장", "삭제", "읽기", "쓰기", "다운로드", "업로드", "file", "save", "delete", "read", "write"],
                "task_types": ["file_management"],
                "output_format": "File operation result",
                "endpoint": DEFAULT_ACP_ENDPOINT
            },
            "memo_agent": {
                "keywords": ["메모", "기록", "노트", "저장", "기억", "memo", "note", "record", "save", "remember"],
                "task_types": ["memo_creation", "note_taking"],
                "output_format": "Memo data",
                "endpoint": DEFAULT_ACP_ENDPOINT
            }
        }

        # Regex pattern definitions (improved version)
        self.task_patterns = [
            # Weather-related
            {
                "pattern": r"(날씨|기온|온도|비|눈|바람|습도|미세먼지|weather|temperature|rain|snow)",
                "agent": "weather_agent",
                "task_type": "weather_inquiry",
                "message_template": "오늘 {location}의 날씨 정보를 자세히 알려주세요. 온도, 습도, 강수확률을 포함해주세요."
            },
            # Finance/exchange rate-related
            {
                "pattern": r"(환율|주가|달러|유로|엔|투자|금융|경제|exchange|rate|dollar|stock|finance)",
                "agent": "finance_agent",
                "task_type": "financial_data",
                "message_template": "{financial_topic}에 대한 최신 금융 정보를 조회해주세요."
            },
            # Calculation-related
            {
                "pattern": r"(계산|수학|더하기|빼기|곱하기|나누기|큰수|작은수|평균|합계|calculate|math|plus|minus)",
                "agent": "calculate_agent",
                "task_type": "calculation",
                "message_template": "다음 계산을 수행해주세요: {expression}"
            },
            # Search-related
            {
                "pattern": r"(검색|찾아|정보|알려|조사|최신|뉴스|웹|인터넷|search|find|information|news|web)",
                "agent": "internet_agent",
                "task_type": "search",
                "message_template": "{topic}에 대한 최신 정보를 검색해주세요."
            },
            # Visualization-related
            {
                "pattern": r"(차트|그래프|시각화|표|그림|도표|chart|graph|visualization|plot)",
                "agent": "chart_agent",
                "task_type": "visualization",
                "message_template": "{data_description}을 바탕으로 시각화 차트를 생성해주세요."
            },
            # Analysis-related
            {
                "pattern": r"(분석|검토|평가|비교|조사|연구|analysis|review|evaluate|compare|research)",
                "agent": "analysis_agent",
                "task_type": "analysis",
                "message_template": "{subject}에 대해 상세한 분석을 수행해주세요."
            },
            # Translation-related
            {
                "pattern": r"(번역|영어|한국어|일본어|중국어|언어|translate|english|korean|japanese|chinese)",
                "agent": "translate_agent",
                "task_type": "translation",
                "message_template": "다음을 {target_language}로 번역해주세요: {text_to_translate}"
            },
            # Image-related
            {
                "pattern": r"(이미지|그림|사진|이미지생성|그려|만들어|image|picture|photo|generate|create)",
                "agent": "image_agent",
                "task_type": "image_generation",
                "message_template": "{image_description}에 대한 이미지를 생성해주세요."
            },
            # Audio-related
            {
                "pattern": r"(음성|소리|오디오|음악|녹음|재생|audio|sound|music|record|play)",
                "agent": "audio_agent",
                "task_type": "audio_processing",
                "message_template": "{audio_description}에 대한 음성 처리를 수행해주세요."
            },
            # File-related
            {
                "pattern": r"(파일|저장|삭제|읽기|쓰기|다운로드|업로드|file|save|delete|read|write)",
                "agent": "file_agent",
                "task_type": "file_management",
                "message_template": "{file_operation}을 수행해주세요."
            },
            # Memo-related
            {
                "pattern": r"(메모|기록|노트|저장|기억|memo|note|record|save|remember)",
                "agent": "memo_agent",
                "task_type": "memo_creation",
                "message_template": "다음 내용을 메모로 저장해주세요: {memo_content}"
            }
        ]
        
        # LLM manager (lazy loading)
        self.llm_manager = None

        logger.info("🚀 Enhanced Query Decomposer initialized - advanced pattern matching and automatic agent registration")
    
    async def decompose_query(self, query: str, available_agents: List[str] = None) -> QueryDecompositionResult:
        """Decompose complex query into individual tasks"""
        try:
            logger.info(f"🔍 Query decomposition started: '{query}'")

            # 1. Basic semantic analysis
            semantic_query = await self._create_semantic_query(query)

            # 2. Advanced LLM-based decomposition (when LLM is available)
            llm_decomposition = []
            if self.llm_manager:
                llm_decomposition = await self._llm_enhanced_decomposition(query, available_agents)

            # 3. Pattern-based decomposition (always runs)
            pattern_decomposition = self._pattern_based_decomposition(query, available_agents)

            # 4. Merge and optimize results
            task_components = self._merge_decompositions(llm_decomposition, pattern_decomposition)

            # 5. Create execution plan
            execution_plan = self._create_execution_plan(task_components)

            # 6. Determine strategy
            execution_strategy = self._determine_execution_strategy(task_components)

            # 7. Estimate time
            estimated_time = self._estimate_execution_time(task_components, execution_strategy)

            # 8. Calculate confidence
            overall_confidence = self._calculate_overall_confidence(task_components)

            # 9. Generate reasoning explanation
            reasoning = self._generate_reasoning(task_components, execution_strategy)

            result = QueryDecompositionResult(
                original_query=query,
                semantic_query=semantic_query,
                task_components=task_components,
                execution_strategy=execution_strategy,
                execution_plan=execution_plan,
                estimated_time=estimated_time,
                overall_confidence=overall_confidence,
                reasoning=reasoning
            )

            logger.info(f"✅ Query decomposition complete: {len(task_components)} tasks, {execution_strategy.value} strategy")
            return result

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return await self._create_fallback_decomposition(query, available_agents)

    async def _create_semantic_query(self, query: str) -> SemanticQuery:
        """Create basic semantic query"""
        try:
            # Simple entity and concept extraction
            entities = self._extract_entities(query)
            concepts = self._extract_concepts(query)
            relations = self._extract_relations(query)

            return SemanticQuery.create_from_text(
                query_text=query,
                intent="multi_task_execution",
                entities=entities,
                concepts=concepts,
                relations=relations
            )
        except Exception as e:
            logger.error(f"SemanticQuery creation failed: {e}")
            return SemanticQuery.create_from_text(query_text=query)
    
    async def _llm_enhanced_decomposition(self, query: str, available_agents: List[str] = None) -> List[TaskComponent]:
        """Advanced LLM-based query decomposition"""
        try:
            available_agents_str = ", ".join(available_agents or list(self.agent_capabilities.keys()))
            
            prompt = f"""
당신은 복잡한 사용자 쿼리를 개별 실행 가능한 태스크로 분해하는 전문가입니다.

**사용자 쿼리:** "{query}"

**사용 가능한 에이전트들:** {available_agents_str}

**태스크 분해 원칙:**
1. 쿼리를 논리적으로 분리 가능한 개별 태스크로 나누기
2. 각 태스크는 하나의 에이전트가 독립적으로 수행 가능해야 함
3. 태스크 간 의존성과 실행 순서 고려
4. 각 에이전트에게 구체적이고 명확한 지시사항 작성

**분석 예시:**
- "오늘 날씨 확인하고 환율 정보도 알려줘" 
  → 태스크1: 날씨 조회 (weather_agent)
  → 태스크2: 환율 조회 (finance_agent)

**응답 형식 (JSON):**
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "task_type": "weather_inquiry",
            "original_text": "오늘 날씨를 확인하고",
            "processed_message": "오늘 서울의 날씨 정보를 자세히 알려주세요. 온도, 습도, 강수확률을 포함해주세요.",
            "target_agent": "weather_agent",
            "priority": 1,
            "dependencies": [],
            "expected_output": "날씨 정보 데이터"
        }}
    ],
    "reasoning": "분해 근거 설명"
}}
"""
            
            result_str = await self.llm_manager.invoke_llm(
                OntologyLLMType.SEMANTIC_ANALYZER, 
                prompt
            )
            
            # Attempt JSON parsing
            result = None
            try:
                result = json.loads(result_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {result_str[:100]}...")

            if result and "tasks" in result:
                tasks = []
                for task_data in result["tasks"]:
                    task = TaskComponent(
                        task_id=task_data.get("task_id", f"task_{uuid.uuid4().hex[:6]}"),
                        task_type=task_data.get("task_type", "general"),
                        original_text=task_data.get("original_text", ""),
                        processed_message=task_data.get("processed_message", ""),
                        target_agent=task_data.get("target_agent", "internet_agent"),
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        expected_output=task_data.get("expected_output", ""),
                        confidence=0.9  # LLM decomposition has high confidence
                    )
                    tasks.append(task)

                logger.info(f"LLM decomposition complete: {len(tasks)} tasks")
                return tasks

        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")

        return []

    def _pattern_based_decomposition(self, query: str, available_agents: List[str] = None) -> List[TaskComponent]:
        """Pattern-based query decomposition (fallback method) - enhanced version"""
        tasks = []

        # Split query by connectors
        segments = self._split_query_by_connectors(query)

        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue

            # Step 1: Regex pattern matching
            matched_pattern = None
            best_confidence = 0.0

            for pattern_config in self.task_patterns:
                if re.search(pattern_config["pattern"], segment, re.IGNORECASE):
                    # Calculate confidence from number of matched keywords
                    keywords_in_pattern = pattern_config["pattern"].split("|")
                    matched_keywords = sum(1 for keyword in keywords_in_pattern if keyword.lower() in segment.lower())
                    confidence = min(0.9, 0.6 + (matched_keywords * 0.1))

                    if confidence > best_confidence:
                        matched_pattern = pattern_config
                        best_confidence = confidence

            # Step 2: Keyword-based agent matching (when pattern matching fails)
            if not matched_pattern:
                matched_agent, keyword_confidence = self._match_agent_by_keywords(segment)
                if matched_agent:
                    # Dynamically create pattern
                    matched_pattern = {
                        "agent": matched_agent,
                        "task_type": self._infer_task_type_from_agent(matched_agent),
                        "message_template": self._get_default_message_template(matched_agent)
                    }
                    best_confidence = keyword_confidence

            if matched_pattern:
                # Verify agent availability
                target_agent = matched_pattern["agent"]
                if available_agents and target_agent not in available_agents:
                    # Find an available alternative agent
                    target_agent = self._find_alternative_agent(matched_pattern["agent"], available_agents)

                # Process message for agent
                processed_message = self._process_message_for_agent(
                    segment,
                    matched_pattern["message_template"],
                    matched_pattern["task_type"]
                )

                task = TaskComponent(
                    task_id=f"task_{i+1}",
                    task_type=matched_pattern["task_type"],
                    original_text=segment,
                    processed_message=processed_message,
                    target_agent=target_agent,
                    priority=i+1,  # Assign priority in order
                    dependencies=[],
                    expected_output=self.agent_capabilities.get(target_agent, {}).get("output_format", "Result data"),
                    confidence=best_confidence
                )

                tasks.append(task)
            else:
                # Final fallback: process as general task
                target_agent = self._get_fallback_agent(available_agents)

                task = TaskComponent(
                    task_id=f"task_{i+1}",
                    task_type="general",
                    original_text=segment,
                    processed_message=f"{segment}에 대해 자세한 정보를 제공해주세요.",
                    target_agent=target_agent,
                    priority=i+1,
                    dependencies=[],
                    expected_output="Requested information",
                    confidence=0.5  # Low confidence
                )
                tasks.append(task)

        logger.info(f"Enhanced pattern-based decomposition complete: {len(tasks)} tasks")
        return tasks

    def _match_agent_by_keywords(self, text: str) -> Tuple[Optional[str], float]:
        """Keyword-based agent matching - enhanced version"""
        text_lower = text.lower()
        best_agent = None
        best_score = 0.0

        # Tokenize text (by whitespace and punctuation)
        import re
        text_tokens = set(re.findall(r'\b\w+\b', text_lower))

        for agent_id, agent_info in self.agent_capabilities.items():
            score = 0.0
            total_keywords = len(agent_info["keywords"])
            matched_keywords = []

            for keyword in agent_info["keywords"]:
                keyword_lower = keyword.lower()

                # Exact word match
                if keyword_lower in text_tokens:
                    score += 2.0  # Higher score for exact match
                    matched_keywords.append(keyword)
                # Substring match
                elif keyword_lower in text_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
                # Fuzzy match (within edit distance 1)
                else:
                    for token in text_tokens:
                        if self._is_similar(keyword_lower, token):
                            score += 0.5
                            matched_keywords.append(f"{keyword}~{token}")
                            break

            # Score adjustment: consider number and quality of matched keywords
            if matched_keywords:
                # Base score (match ratio)
                base_score = score / (total_keywords * 2)  # Normalize to max value

                # Bonus score: additional points for multiple keyword matches
                bonus_score = min(0.3, len(matched_keywords) * 0.1)

                # Text length consideration: shorter text with match = higher confidence
                length_bonus = max(0, (50 - len(text)) / 100 * 0.2)

                final_score = base_score + bonus_score + length_bonus

                if final_score > best_score:
                    best_agent = agent_id
                    best_score = final_score

        # Confidence calculation (more refined method)
        if best_score > 0:
            confidence = min(0.9, 0.4 + best_score * 0.6)
        else:
            confidence = 0.0

        return best_agent, confidence

    def _is_similar(self, word1: str, word2: str, max_distance: int = 1) -> bool:
        """Check word similarity (edit distance based)"""
        if len(word1) < 3 or len(word2) < 3:
            return False

        # Simple edit distance check
        if abs(len(word1) - len(word2)) > max_distance:
            return False

        # Check common prefix/suffix
        if word1.startswith(word2[:3]) or word2.startswith(word1[:3]):
            return True

        if word1.endswith(word2[-3:]) or word2.endswith(word1[-3:]):
            return True

        return False

    def _infer_task_type_from_agent(self, agent_id: str) -> str:
        """Infer task type from agent ID"""
        agent_to_task_type = {
            "weather_agent": "weather_inquiry",
            "finance_agent": "financial_data",
            "calculate_agent": "calculation",
            "internet_agent": "search",
            "chart_agent": "visualization",
            "analysis_agent": "analysis",
            "translate_agent": "translation",
            "image_agent": "image_generation",
            "audio_agent": "audio_generation",
            "file_agent": "file_management",
            "memo_agent": "memo_creation"
        }
        return agent_to_task_type.get(agent_id, "general")
    
    def _get_default_message_template(self, agent_id: str) -> str:
        """Default message template per agent"""
        templates = {
            "weather_agent": "오늘 {location}의 날씨 정보를 자세히 알려주세요. 온도, 습도, 강수확률을 포함해주세요.",
            "finance_agent": "{financial_topic}에 대한 금융 정보를 조회해주세요. 최신 데이터와 분석을 포함해주세요.",
            "calculate_agent": "다음 계산을 수행해주세요: {expression}. 계산 과정과 결과를 명확히 보여주세요.",
            "internet_agent": "{topic}에 대한 최신 정보를 검색해주세요. 신뢰할 수 있는 출처의 정보를 우선적으로 수집해주세요.",
            "chart_agent": "{data_description}을 바탕으로 시각화 차트를 생성해주세요.",
            "analysis_agent": "{subject}에 대해 상세한 분석을 수행해주세요. 핵심 포인트와 인사이트를 제공해주세요.",
            "translate_agent": "다음 텍스트를 {target_language}로 번역해주세요: {text_to_translate}. 자연스럽고 정확한 번역을 제공해주세요.",
            "image_agent": "{image_description}에 대한 이미지를 생성해주세요. 고품질의 결과물을 제공해주세요.",
            "audio_agent": "{audio_description}에 대한 음성 처리를 수행해주세요. 고품질의 오디오 결과를 제공해주세요.",
            "file_agent": "{file_operation}을 수행해주세요. 파일을 안전하게 처리하고 결과를 보고해주세요.",
            "memo_agent": "다음 내용을 메모로 저장해주세요: {memo_content}. 체계적으로 정리하여 보관해주세요."
        }
        return templates.get(agent_id, "{original_text}에 대해 자세한 정보를 제공해주세요.")
    
    def _find_alternative_agent(self, preferred_agent: str, available_agents: List[str]) -> str:
        """Find alternative agent when preferred agent is unavailable"""
        # Mapping of agents with similar capabilities
        alternatives = {
            "weather_agent": ["internet_agent"],
            "finance_agent": ["internet_agent"],
            "calculate_agent": ["internet_agent"],
            "translate_agent": ["internet_agent"],
            "image_agent": ["internet_agent"],
            "audio_agent": ["internet_agent"],
            "file_agent": ["internet_agent"],
            "chart_agent": ["analysis_agent", "internet_agent"],
            "analysis_agent": ["internet_agent"],
            "memo_agent": ["internet_agent"]
        }
        
        if preferred_agent in alternatives:
            for alt_agent in alternatives[preferred_agent]:
                if alt_agent in available_agents:
                    return alt_agent
        
        return self._get_fallback_agent(available_agents)
    
    def _get_fallback_agent(self, available_agents: List[str]) -> str:
        """Select final fallback agent"""
        if available_agents:
            # Prefer internet_agent as first choice
            if "internet_agent" in available_agents:
                return "internet_agent"
            else:
                return available_agents[0]
        else:
            return "internet_agent"

    def _split_query_by_connectors(self, query: str) -> List[str]:
        """Split query by connectors - enhanced version"""
        # Korean and English connector patterns
        connectors = [
            r"그리고|그리구|그리고서",
            r"또한|또|또는|혹은",
            r"다음에|그다음|그 다음",
            r"이후에|그 후에|그후",
            r"마지막으로|끝으로",
            r"하고|해서|하여",
            r"and|then|after|also|next",  # English connectors
            r",\s*",  # Comma
            r";\s*",  # Semicolon
        ]

        # Combine all connectors into one pattern
        combined_pattern = "|".join(f"({pattern})" for pattern in connectors)

        # Split
        segments = re.split(combined_pattern, query, flags=re.IGNORECASE)

        # Extract actual text only (remove connector tokens)
        cleaned_segments = []
        for segment in segments:
            if segment and segment.strip() and not re.match(combined_pattern, segment.strip(), re.IGNORECASE):
                # Add only non-empty, non-connector segments
                clean_text = segment.strip()
                if len(clean_text) > 2:  # Exclude very short text
                    cleaned_segments.append(clean_text)

        # Try manual splitting if auto-split failed
        if len(cleaned_segments) <= 1 and len(query) > 20:
            # Try special patterns
            manual_patterns = [
                r"(.+?)을?\s+확인하[고|하](.+)",
                r"(.+?)를?\s+하[고|하](.+)",
                r"(.+?)하[고|하]\s*(.+)",
                r"(.+?)\s+and\s+(.+)"
            ]

            for pattern in manual_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    cleaned_segments = [match.group(1).strip(), match.group(2).strip()]
                    break

        return cleaned_segments if cleaned_segments else [query]

    def _process_message_for_agent(self, original_text: str, template: str, task_type: str) -> str:
        """Process message per agent - language-aware enhanced version"""
        try:
            # Detect language
            detected_language = self._detect_language(original_text)
            
            if task_type == "weather_inquiry":
                # 지역 추출 시도
                location = self._extract_location(original_text)
                if not location:
                    location = "Seoul" if detected_language == "en" else "서울"
                return template.format(location=location)
            
            elif task_type == "currency_inquiry":
                # 통화 추출 시도
                currency = self._extract_currency(original_text)
                if not currency:
                    currency = "USD" if detected_language == "en" else "달러"
                return template.format(currency=currency)
            
            elif task_type == "calculation":
                # Extract calculation expression
                expression = self._extract_calculation(original_text)
                if not expression:
                    expression = "simple calculation" if detected_language == "en" else "간단한 계산"
                return template.format(expression=expression)

            elif task_type == "search":
                # Extract search topic
                topic = self._extract_search_topic(original_text)
                if not topic:
                    topic = "requested information" if detected_language == "en" else "요청된 정보"
                return template.format(topic=topic)

            elif task_type in ["stock_inquiry", "crypto_inquiry", "financial_data"]:
                # Extract financial instrument/topic
                financial_topic = self._extract_financial_topic(original_text)
                if not financial_topic:
                    financial_topic = "financial information" if detected_language == "en" else "금융 정보"

                if task_type == "stock_inquiry":
                    stock_name = self._extract_stock_name(original_text) or financial_topic
                    return template.format(stock_name=stock_name)
                elif task_type == "crypto_inquiry":
                    crypto_name = self._extract_crypto_name(original_text) or financial_topic
                    return template.format(crypto_name=crypto_name)
                else:
                    return template.format(financial_topic=financial_topic)

            elif task_type == "translation":
                # Extract target language and text to translate
                target_language = self._extract_target_language(original_text)
                if not target_language:
                    # Determine translation direction based on detected language
                    target_language = "Korean" if detected_language == "en" else "English"

                text_to_translate = self._extract_text_to_translate(original_text) or original_text
                return template.format(target_language=target_language, text_to_translate=text_to_translate)

            elif task_type == "image_generation":
                # Extract image description
                image_description = self._extract_image_description(original_text) or original_text
                return template.format(image_description=image_description)

            elif task_type == "audio_generation":
                # Extract audio description
                audio_description = self._extract_audio_description(original_text) or original_text
                return template.format(audio_description=audio_description)

            elif task_type == "file_management":
                # Extract file operation
                file_operation = self._extract_file_operation(original_text) or original_text
                return template.format(file_operation=file_operation)

            elif task_type == "memo_creation":
                # Extract memo content
                memo_content = self._extract_memo_content(original_text) or original_text
                return template.format(memo_content=memo_content)

            elif task_type == "math_operation":
                # Extract math operation
                math_operation = self._extract_math_operation(original_text) or original_text
                return template.format(math_operation=math_operation)

            elif task_type == "visualization":
                # Extract data description
                data_description = self._extract_data_description(original_text)
                if not data_description:
                    data_description = "collected data" if detected_language == "en" else "수집된 데이터"
                return template.format(data_description=data_description)

            elif task_type == "analysis":
                # Extract analysis subject
                subject = self._extract_analysis_subject(original_text)
                if not subject:
                    subject = "topic" if detected_language == "en" else "주제"
                return template.format(subject=subject)

            else:
                # Default processing - adjust message by language
                if detected_language == "en":
                    return f"Please provide detailed information about {original_text}."
                else:
                    return f"{original_text}에 대해 자세한 정보를 제공해주세요."

        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            if self._detect_language(original_text) == "en":
                return f"Please process {original_text}."
            else:
                return f"{original_text}에 대해 처리해주세요."

    def _detect_language(self, text: str) -> str:
        """Simple language detection (Korean/English)"""
        # Calculate ratio of Korean characters
        korean_chars = len([c for c in text if '\uAC00' <= c <= '\uD7A3' or '\u3131' <= c <= '\u318E'])
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "en"  # Default value

        korean_ratio = korean_chars / total_chars

        # Treat as Korean if Korean character ratio >= 30%
        if korean_ratio >= 0.3:
            return "ko"
        else:
            return "en"

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location name from text - Korean/English support"""
        # Korean city names
        korean_locations = ["서울", "부산", "대구", "인천", "대전", "광주", "울산", "세종", "경기", "강원", "제주"]

        # English city names
        english_locations = [
            "seoul", "busan", "daegu", "incheon", "daejeon", "gwangju", "ulsan", "sejong",
            "tokyo", "osaka", "kyoto", "new york", "london", "paris", "berlin", "madrid",
            "rome", "beijing", "shanghai", "moscow", "sydney", "toronto", "vancouver"
        ]

        text_lower = text.lower()

        # Check Korean city names
        for location in korean_locations:
            if location in text:
                return location

        # Check English city names
        for location in english_locations:
            if location in text_lower:
                return location.title()  # Capitalize first letter

        return None

    def _extract_currency(self, text: str) -> Optional[str]:
        """Extract currency name from text"""
        currencies = {
            "달러": "USD", "유로": "EUR", "엔": "JPY", "위안": "CNY",
            "파운드": "GBP", "원": "KRW"
        }
        for currency, code in currencies.items():
            if currency in text:
                return currency
        return None
    
    def _extract_calculation(self, text: str) -> Optional[str]:
        """Extract calculation expression from text"""
        # Simple math expression pattern matching
        math_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)'
        match = re.search(math_pattern, text)
        if match:
            return match.group(0)

        # Percentage calculation
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        if re.search(percent_pattern, text):
            return "퍼센트 계산"

        return None

    def _extract_search_topic(self, text: str) -> Optional[str]:
        """Extract search topic from text"""
        # Patterns: "~에 대한", "~관련", "~정보"
        patterns = [
            r'(.+?)에\s*대한',
            r'(.+?)\s*관련',
            r'(.+?)\s*정보'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _extract_data_description(self, text: str) -> Optional[str]:
        """Extract data description from text"""
        if "차트" in text or "그래프" in text:
            return "수집된 데이터"
        return None

    def _extract_analysis_subject(self, text: str) -> Optional[str]:
        """Extract analysis subject from text"""
        if "분석" in text:
            # Patterns: "~을 분석", "~에 대한 분석"
            patterns = [
                r'(.+?)을\s*분석',
                r'(.+?)를\s*분석',
                r'(.+?)\s*분석'
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()

        return None

    def _extract_financial_topic(self, text: str) -> Optional[str]:
        """Extract financial topic from text"""
        financial_keywords = {
            "주식": "주식 시장",
            "코인": "암호화폐",
            "비트코인": "비트코인",
            "이더리움": "이더리움",
            "금리": "금리",
            "채권": "채권",
            "펀드": "펀드",
            "ETF": "ETF",
            "원자재": "원자재"
        }
        
        for keyword, topic in financial_keywords.items():
            if keyword in text:
                return topic
        
        return None
    
    def _extract_stock_name(self, text: str) -> Optional[str]:
        """Extract stock name from text"""
        # Major Korean stocks
        stocks = ["삼성전자", "SK하이닉스", "네이버", "카카오", "LG화학", "현대차", "KB금융", "신한지주"]
        for stock in stocks:
            if stock in text:
                return stock

        # Indices like KOSPI, NASDAQ, Dow
        indices = ["코스피", "나스닥", "다우", "S&P500"]
        for index in indices:
            if index in text:
                return index

        return None

    def _extract_crypto_name(self, text: str) -> Optional[str]:
        """Extract cryptocurrency name from text"""
        cryptos = {
            "비트코인": "Bitcoin",
            "이더리움": "Ethereum", 
            "리플": "Ripple",
            "에이다": "Cardano",
            "도지코인": "Dogecoin"
        }
        
        for crypto_kr, crypto_en in cryptos.items():
            if crypto_kr in text or crypto_en.lower() in text.lower():
                return crypto_kr
        
        return None
    
    def _extract_target_language(self, text: str) -> Optional[str]:
        """Extract target translation language from text"""
        languages = {
            "영어": "영어",
            "한국어": "한국어",
            "중국어": "중국어",
            "일본어": "일본어",
            "독일어": "독일어",
            "프랑스어": "프랑스어",
            "스페인어": "스페인어"
        }
        
        for lang in languages.keys():
            if f"{lang}로" in text or f"{lang}로 번역" in text:
                return lang
        
        return None
    
    def _extract_text_to_translate(self, text: str) -> Optional[str]:
        """Extract text to translate - enhanced version"""
        # Extract text to translate using various patterns
        patterns = [
            r'["\'](.+?)["\'].*번역',  # 따옴표로 감싼 텍스트
            r'(.+?)을\s*번역',
            r'(.+?)를\s*번역',
            r'번역.*["\'](.+?)["\']',  # 번역 뒤에 오는 따옴표 텍스트
            r'translate.*["\'](.+?)["\']',  # 영어 패턴
            r'["\'](.+?)["\'].*translate',
            r'(.+?)\s+to\s+\w+',  # "text to Korean" 패턴
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Exclude results that are too short or are just commands
                if len(extracted) > 2 and not re.match(r'^(번역|translate|을|를|해|줘)$', extracted):
                    return extracted

        # When pattern matching fails, remove command words from full text
        cleaned_text = re.sub(r'(번역|translate|해줘|해|줘|을|를)', '', text, flags=re.IGNORECASE).strip()
        if len(cleaned_text) > 5:
            return cleaned_text
        
        return None
    
    def _extract_image_description(self, text: str) -> Optional[str]:
        """Extract image description from text - enhanced version"""
        # Extract image description using various patterns
        patterns = [
            r'(.+?)\s*그림',
            r'(.+?)\s*이미지', 
            r'(.+?)\s*사진',
            r'(.+?)\s+그려',
            r'(.+?)\s+draw',
            r'(.+?)\s+image',
            r'(.+?)\s+picture',
            r'create\s+(.+?)\s+(image|picture)',
            r'generate\s+(.+?)\s+(image|picture)',
            r'그려.*?([가-힣\w\s]+)',  # "그려줘" 뒤의 설명
            r'만들.*?([가-힣\w\s]+)\s*(그림|이미지|사진)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Verify it's actual description, not a command word
                if len(extracted) > 1 and not re.match(r'^(그려|그림|이미지|사진|만들|생성|create|draw|generate)$', extracted, re.IGNORECASE):
                    return extracted

        # Keyword-based extraction
        image_keywords = ["그려", "생성", "만들어", "create", "draw", "generate", "paint"]
        if any(keyword in text.lower() for keyword in image_keywords):
            # Extract remainder after removing command words
            cleaned_text = text
            for keyword in ["그려줘", "그려", "만들어줘", "만들어", "생성해줘", "생성해", "create", "draw", "generate"]:
                cleaned_text = re.sub(rf'\b{keyword}\b', '', cleaned_text, flags=re.IGNORECASE)

            cleaned_text = cleaned_text.strip()
            if len(cleaned_text) > 2:
                return cleaned_text

            # Use full text as description
            return text

        return None

    def _extract_audio_description(self, text: str) -> Optional[str]:
        """Extract audio description from text"""
        # Audio-related keywords
        if any(keyword in text for keyword in ["읽어", "말해", "소리", "음성", "재생"]):
            return text

        return None

    def _extract_file_operation(self, text: str) -> Optional[str]:
        """Extract file operation from text"""
        operations = {
            "다운로드": "파일 다운로드",
            "업로드": "파일 업로드",
            "저장": "파일 저장",
            "불러오기": "파일 불러오기",
            "열기": "파일 열기",
            "읽기": "파일 읽기"
        }

        for op_kr, op_desc in operations.items():
            if op_kr in text:
                return op_desc

        return None

    def _extract_memo_content(self, text: str) -> Optional[str]:
        """Extract memo content from text"""
        # Patterns: "~을 메모", "~를 저장"
        patterns = [
            r'(.+?)을\s*메모',
            r'(.+?)를\s*메모',
            r'(.+?)을\s*저장',
            r'(.+?)를\s*저장'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _extract_math_operation(self, text: str) -> Optional[str]:
        """Extract math operation from text"""
        math_keywords = {
            "평균": "평균 계산",
            "합계": "합계 계산",
            "총합": "총합 계산",
            "퍼센트": "퍼센트 계산",
            "백분율": "백분율 계산",
            "제곱": "제곱 계산",
            "루트": "제곱근 계산"
        }

        for keyword, operation in math_keywords.items():
            if keyword in text:
                return operation

        return None

    def _extract_entities(self, query: str) -> List[str]:
        """Simple entity extraction"""
        entities = []

        # Number patterns
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        entities.extend([f"number_{num}" for num in numbers])

        # Time-related words
        time_words = ["오늘", "내일", "어제", "지금", "현재"]
        for word in time_words:
            if word in query:
                entities.append(f"time_{word}")

        return list(set(entities))

    def _extract_concepts(self, query: str) -> List[str]:
        """Simple concept extraction"""
        concepts = []

        # Domain concepts
        domain_concepts = {
            "날씨": "weather_domain",
            "환율": "finance_domain",
            "계산": "calculation_domain",
            "정보": "information_domain"
        }

        for keyword, concept in domain_concepts.items():
            if keyword in query:
                concepts.append(concept)

        return concepts

    def _extract_relations(self, query: str) -> List[str]:
        """Simple relation extraction"""
        relations = []

        if "그리고" in query or "," in query:
            relations.append("sequential_execution")

        if "비교" in query:
            relations.append("comparison")

        return relations

    def _merge_decompositions(self, llm_tasks: List[TaskComponent], pattern_tasks: List[TaskComponent]) -> List[TaskComponent]:
        """Merge LLM and pattern-based decomposition results"""
        # Prefer LLM results; supplement missing parts with pattern results
        if llm_tasks:
            return llm_tasks

        return pattern_tasks

    def _create_execution_plan(self, tasks: List[TaskComponent]) -> List[List[str]]:
        """Create execution plan (parallel groups per step)"""
        if not tasks:
            return []

        # If no dependencies, run all tasks in parallel
        if not any(task.dependencies for task in tasks):
            return [[task.task_id for task in tasks]]

        # Build levels based on dependencies
        levels = []
        remaining_tasks = tasks.copy()
        completed_tasks = set()

        while remaining_tasks:
            current_level = []

            # Find tasks whose dependencies are all resolved
            for task in remaining_tasks[:]:
                dependencies_met = all(dep in completed_tasks for dep in task.dependencies)

                if dependencies_met:
                    current_level.append(task.task_id)
                    completed_tasks.add(task.task_id)
                    remaining_tasks.remove(task)

            if current_level:
                levels.append(current_level)
            else:
                # Force-add when circular dependencies prevent progress
                if remaining_tasks:
                    levels.append([remaining_tasks[0].task_id])
                    completed_tasks.add(remaining_tasks[0].task_id)
                    remaining_tasks.pop(0)

        return levels
    
    def _determine_execution_strategy(self, tasks: List[TaskComponent]) -> ExecutionStrategy:
        """Determine execution strategy"""
        if len(tasks) <= 1:
            return ExecutionStrategy.SINGLE_AGENT

        # Analyze dependencies
        has_dependencies = any(task.dependencies for task in tasks)

        if has_dependencies:
            return ExecutionStrategy.SEQUENTIAL

        # Strategy based on task count (no dependencies)
        if len(tasks) <= 3:
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.HYBRID

    def _estimate_execution_time(self, tasks: List[TaskComponent], strategy: ExecutionStrategy) -> float:
        """Estimate execution time"""
        if not tasks:
            return 0.0

        # Base time per task type (seconds)
        base_times = {
            "weather_inquiry": 3.0,
            "currency_inquiry": 2.0,
            "stock_inquiry": 3.0,
            "crypto_inquiry": 3.0,
            "financial_data": 4.0,
            "calculation": 1.0,
            "math_operation": 2.0,
            "search": 5.0,
            "visualization": 4.0,
            "analysis": 6.0,
            "translation": 3.0,
            "image_generation": 8.0,
            "image_editing": 6.0,
            "image_processing": 5.0,
            "audio_generation": 7.0,
            "audio_processing": 5.0,
            "voice_synthesis": 4.0,
            "file_management": 3.0,
            "document_processing": 4.0,
            "file_operation": 2.0,
            "memo_creation": 2.0,
            "data_storage": 2.0,
            "record_keeping": 3.0,
            "general": 3.0
        }
        
        if strategy == ExecutionStrategy.PARALLEL:
            # Parallel: time of the slowest task
            return max(base_times.get(task.task_type, 3.0) for task in tasks)
        else:
            # Sequential: sum of all task times
            return sum(base_times.get(task.task_type, 3.0) for task in tasks)

    def _calculate_overall_confidence(self, tasks: List[TaskComponent]) -> float:
        """Calculate overall confidence score"""
        if not tasks:
            return 0.0

        return sum(task.confidence for task in tasks) / len(tasks)

    def _generate_reasoning(self, tasks: List[TaskComponent], strategy: ExecutionStrategy) -> str:
        """Generate decomposition reasoning explanation"""
        task_summary = ", ".join([f"{task.task_type}({task.target_agent})" for task in tasks])

        return f"""
Query decomposed into {len(tasks)} individual tasks:
- Tasks: {task_summary}
- Execution strategy: {strategy.value}
- Each task is processed independently by the corresponding specialist agent.
"""

    async def _create_fallback_decomposition(self, query: str, available_agents: List[str] = None) -> QueryDecompositionResult:
        """Create fallback decomposition result"""
        logger.warning("Creating fallback decomposition result")

        # Process as single default task
        task = TaskComponent(
            task_id="fallback_task",
            task_type="general",
            original_text=query,
            processed_message=f"다음 요청을 처리해주세요: {query}",
            target_agent="internet_agent",
            priority=1,
            dependencies=[],
            expected_output="Requested information",
            confidence=0.5
        )

        semantic_query = SemanticQuery.create_from_text(query_text=query)

        return QueryDecompositionResult(
            original_query=query,
            semantic_query=semantic_query,
            task_components=[task],
            execution_strategy=ExecutionStrategy.SINGLE_AGENT,
            execution_plan=[["fallback_task"]],
            estimated_time=5.0,
            overall_confidence=0.5,
            reasoning="Fallback mode: processing as single task"
        )


# Convenience functions
async def decompose_user_query(query: str, available_agents: List[str] = None) -> QueryDecompositionResult:
    """Convenience function for user query decomposition"""
    decomposer = EnhancedQueryDecomposer()
    return await decomposer.decompose_query(query, available_agents)


def get_agent_specific_message(task: TaskComponent) -> str:
    """Extract agent-specific message from task"""
    return task.processed_message


def get_execution_order(decomposition_result: QueryDecompositionResult) -> List[Tuple[str, str]]:
    """Return execution order list (task_id, agent_id)"""
    execution_order = []

    for level in decomposition_result.execution_plan:
        for task_id in level:
            # Find task with matching task_id
            task = next((t for t in decomposition_result.task_components if t.task_id == task_id), None)
            if task:
                execution_order.append((task_id, task.target_agent))
    
    return execution_order


logger.info("🧠 고도화된 쿼리 분해기 로드 완료!") 