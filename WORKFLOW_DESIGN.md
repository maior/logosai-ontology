# 🌊 워크플로우 설계 원리

## 📋 개요

LogosAI 온톨로지 시스템의 워크플로우 설계는 LLM 기반 지능적 분석을 통해 사용자 쿼리에 최적화된 에이전트 조합과 실행 순서를 동적으로 결정합니다.

## 🎯 설계 원칙

### 1. 의미론적 분석 우선
- 사용자 쿼리의 의도와 맥락을 정확히 파악
- 도메인별 전문 지식 활용
- 복잡도에 따른 적응적 접근

### 2. 최적화 전략 자동 선택
- 단일/순차/병렬/하이브리드 전략 지능적 결정
- 에이전트 간 의존성 분석
- 성능과 품질의 균형

### 3. 동적 적응성
- 실행 중 상황 변화에 대응
- 실패 시 대안 경로 제공
- 학습 기반 지속적 개선

## 🧠 LLM 기반 워크플로우 설계

### 설계 프롬프트 시스템
```python
workflow_design_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 실제 성공 사례와 검증된 워크플로우 패턴을 바탕으로 
    최적의 에이전트 워크플로우를 설계하는 전문가입니다.

    **📚 검증된 워크플로우 패턴 라이브러리:**

    🔍 **패턴 1: 정보수집→분석→시각화→보고서** (복잡도: 복잡)
    - 적용 케이스: ESG 경영 분석, 기술 트렌드 조사, 시장 연구
    - 단계 구성: internet_agent → analysis_agent → chart_agent
    - 실제 예시: "글로벌 ESG 트렌드를 분석하고 벤치마킹 보고서 작성"

    📊 **패턴 2: 조회→계산→정리** (복잡도: 보통) 🚀 **병렬 최적화 가능**
    - 적용 케이스: 환율 계산, 투자 수익률, 금융 분석
    - 단계 구성: finance_agent → calculate_agent → chart_agent
    - **병렬 최적화**: 다중 통화 조회 시 각 통화별 병렬 처리

    🌤️ **패턴 3: 예보→계획→스케줄링** (복잡도: 복잡)
    - 적용 케이스: 여행 계획, 일정 관리, 출퇴근 최적화
    - 단계 구성: weather_agent → internet_agent → calendar_agent

    **🚀 병렬 처리 최적화 가이드라인:**
    
    **💡 병렬 처리 감지 키워드:**
    - **복수 개체**: "A와 B", "여러", "각각", "다중", "모든"
    - **비교 키워드**: "비교", "대비", "차이", "vs", "versus"
    - **독립 작업**: "별도로", "개별적으로", "동시에", "함께"
    """),
    
    ("human", """다음 쿼리에 대한 최적의 워크플로우를 설계해주세요:

    **사용자 쿼리:** "{original_query}"
    **사용 가능한 에이전트:** {available_agents}

    다음 JSON 형식으로 응답해주세요:

    {{
        "workflow_strategy": "SINGLE_AGENT|SEQUENTIAL|PARALLEL|HYBRID",
        "complexity_level": "simple|moderate|complex|sophisticated",
        "steps": [
            {{
                "step_id": "step_1",
                "agent_id": "agent_name",
                "semantic_purpose": "이 단계의 의미론적 목적",
                "estimated_time": 30.0,
                "depends_on": [],
                "parallel_group": null,
                "confidence": 0.9
            }}
        ],
        "execution_flow": {{
            "type": "sequential|parallel|hybrid",
            "parallel_groups": [
                {{
                    "group_id": "group_1",
                    "steps": ["step_1", "step_2"],
                    "execution_type": "parallel"
                }}
            ]
        }},
        "reasoning": "설계 근거와 최적화 전략",
        "estimated_total_time": 90.0,
        "success_probability": 0.95
    }}""")
])
```

## 📊 워크플로우 패턴 라이브러리

### 🔍 패턴 1: 정보수집→분석→시각화
```python
pattern_1 = {
    "name": "정보수집→분석→시각화",
    "complexity": "complex",
    "use_cases": ["ESG 분석", "시장 조사", "트렌드 분석"],
    "steps": [
        {
            "step_id": "data_collection",
            "agent_type": "internet_agent",
            "purpose": "최신 정보 수집",
            "estimated_time": 45.0
        },
        {
            "step_id": "analysis",
            "agent_type": "analysis_agent", 
            "purpose": "데이터 분석 및 인사이트 도출",
            "depends_on": ["data_collection"],
            "estimated_time": 60.0
        },
        {
            "step_id": "visualization",
            "agent_type": "chart_agent",
            "purpose": "시각화 및 보고서 생성",
            "depends_on": ["data_collection", "analysis"],
            "estimated_time": 30.0
        }
    ],
    "total_time": 135.0,
    "success_rate": 0.92
}
```

### 📊 패턴 2: 병렬 최적화 가능한 조회→계산
```python
pattern_2_parallel = {
    "name": "병렬_조회→계산→정리",
    "complexity": "moderate",
    "optimization": "parallel",
    "use_cases": ["다중 환율 조회", "여러 주식 분석", "복수 데이터 비교"],
    "steps": [
        {
            "step_id": "parallel_data_fetch",
            "parallel_group": "fetch_group",
            "sub_steps": [
                {
                    "step_id": "fetch_usd",
                    "agent_type": "currency_agent",
                    "purpose": "USD 환율 조회",
                    "estimated_time": 20.0
                },
                {
                    "step_id": "fetch_eur", 
                    "agent_type": "currency_agent",
                    "purpose": "EUR 환율 조회",
                    "estimated_time": 20.0
                }
            ]
        },
        {
            "step_id": "calculation",
            "agent_type": "calculator_agent",
            "purpose": "환전 금액 계산",
            "depends_on": ["fetch_usd", "fetch_eur"],
            "estimated_time": 15.0
        },
        {
            "step_id": "visualization",
            "agent_type": "chart_agent",
            "purpose": "비교 차트 생성",
            "depends_on": ["calculation"],
            "estimated_time": 25.0
        }
    ],
    "execution_strategy": "hybrid",
    "total_time": 60.0,  # 병렬 처리로 시간 단축
    "success_rate": 0.95
}
```

### 🌤️ 패턴 3: 조건부 워크플로우
```python
pattern_3_conditional = {
    "name": "조건부_예보→계획→실행",
    "complexity": "complex",
    "use_cases": ["여행 계획", "이벤트 기획", "조건부 작업"],
    "steps": [
        {
            "step_id": "condition_check",
            "agent_type": "weather_agent",
            "purpose": "조건 확인 (날씨 등)",
            "estimated_time": 20.0
        },
        {
            "step_id": "conditional_planning",
            "agent_type": "planning_agent",
            "purpose": "조건에 따른 계획 수립",
            "depends_on": ["condition_check"],
            "conditional_logic": {
                "if_sunny": ["outdoor_activities"],
                "if_rainy": ["indoor_activities"]
            },
            "estimated_time": 40.0
        },
        {
            "step_id": "execution",
            "agent_type": "calendar_agent",
            "purpose": "계획 실행 및 스케줄링",
            "depends_on": ["conditional_planning"],
            "estimated_time": 30.0
        }
    ],
    "total_time": 90.0,
    "success_rate": 0.88
}
```

## ⚡ 실행 전략

### 1. SINGLE_AGENT 전략
```python
class SingleAgentStrategy:
    """단일 에이전트 전략"""
    
    @staticmethod
    def is_applicable(query_analysis: Dict) -> bool:
        return (
            query_analysis["complexity"] == "simple" and
            len(query_analysis["required_capabilities"]) == 1 and
            query_analysis["domain_specificity"] > 0.8
        )
    
    async def execute(self, workflow_plan: OntologyWorkflowPlan) -> List[AgentExecutionResult]:
        # 단일 에이전트로 직접 실행
        primary_step = workflow_plan.steps[0]
        result = await self._execute_single_step(primary_step)
        return [result]
```

### 2. SEQUENTIAL 전략
```python
class SequentialStrategy:
    """순차 실행 전략"""
    
    @staticmethod
    def is_applicable(query_analysis: Dict) -> bool:
        return (
            query_analysis["has_dependencies"] and
            query_analysis["parallel_potential"] < 0.3
        )
    
    async def execute(self, workflow_plan: OntologyWorkflowPlan) -> List[AgentExecutionResult]:
        results = []
        context = {}
        
        for step in workflow_plan.steps:
            # 이전 단계 결과를 컨텍스트로 전달
            step_context = self._prepare_step_context(step, context)
            result = await self._execute_step_with_context(step, step_context)
            
            results.append(result)
            context[step.step_id] = result.data
            
        return results
```

### 3. PARALLEL 전략
```python
class ParallelStrategy:
    """병렬 실행 전략"""
    
    @staticmethod
    def is_applicable(query_analysis: Dict) -> bool:
        return (
            query_analysis["parallel_potential"] > 0.7 and
            query_analysis["independent_tasks"] > 1
        )
    
    async def execute(self, workflow_plan: OntologyWorkflowPlan) -> List[AgentExecutionResult]:
        # 병렬 그룹 식별
        parallel_groups = self._identify_parallel_groups(workflow_plan.steps)
        
        results = []
        for group in parallel_groups:
            if group["type"] == "parallel":
                # 병렬 실행
                group_results = await asyncio.gather(*[
                    self._execute_step(step) for step in group["steps"]
                ])
                results.extend(group_results)
            else:
                # 순차 실행
                for step in group["steps"]:
                    result = await self._execute_step(step)
                    results.append(result)
        
        return results
```

### 4. HYBRID 전략
```python
class HybridStrategy:
    """하이브리드 전략 (순차 + 병렬)"""
    
    @staticmethod
    def is_applicable(query_analysis: Dict) -> bool:
        return (
            query_analysis["complexity"] in ["complex", "sophisticated"] and
            query_analysis["mixed_dependencies"] and
            query_analysis["parallel_potential"] > 0.4
        )
    
    async def execute(self, workflow_plan: OntologyWorkflowPlan) -> List[AgentExecutionResult]:
        execution_graph = workflow_plan.execution_graph
        results = []
        completed_steps = set()
        
        while len(completed_steps) < len(workflow_plan.steps):
            # 실행 가능한 단계들 찾기
            ready_steps = self._find_ready_steps(workflow_plan.steps, completed_steps, execution_graph)
            
            # 병렬 실행 가능한 그룹과 순차 실행 단계 분리
            parallel_steps, sequential_steps = self._separate_execution_types(ready_steps)
            
            # 병렬 실행
            if parallel_steps:
                parallel_results = await asyncio.gather(*[
                    self._execute_step(step) for step in parallel_steps
                ])
                results.extend(parallel_results)
                completed_steps.update(step.step_id for step in parallel_steps)
            
            # 순차 실행
            for step in sequential_steps:
                result = await self._execute_step(step)
                results.append(result)
                completed_steps.add(step.step_id)
        
        return results
```

## 🔄 동적 워크플로우 조정

### 실행 중 적응
```python
class DynamicWorkflowAdjuster:
    """동적 워크플로우 조정기"""
    
    async def monitor_and_adjust(self, workflow_plan: OntologyWorkflowPlan, 
                               execution_context: Dict) -> OntologyWorkflowPlan:
        """실행 중 워크플로우 모니터링 및 조정"""
        
        adjustments = []
        
        # 성능 모니터링
        performance_issues = self._detect_performance_issues(execution_context)
        if performance_issues:
            adjustments.extend(self._create_performance_adjustments(performance_issues))
        
        # 실패 감지 및 대응
        failures = self._detect_failures(execution_context)
        if failures:
            adjustments.extend(self._create_failure_recovery_adjustments(failures))
        
        # 품질 모니터링
        quality_issues = self._detect_quality_issues(execution_context)
        if quality_issues:
            adjustments.extend(self._create_quality_adjustments(quality_issues))
        
        # 조정 적용
        if adjustments:
            adjusted_plan = self._apply_adjustments(workflow_plan, adjustments)
            return adjusted_plan
        
        return workflow_plan
    
    def _create_performance_adjustments(self, issues: List[Dict]) -> List[Dict]:
        """성능 문제 해결을 위한 조정"""
        adjustments = []
        
        for issue in issues:
            if issue["type"] == "slow_agent":
                # 느린 에이전트를 더 빠른 대안으로 교체
                adjustments.append({
                    "type": "agent_replacement",
                    "original_agent": issue["agent_id"],
                    "replacement_agent": issue["alternative_agent"],
                    "reason": "performance_optimization"
                })
            elif issue["type"] == "sequential_bottleneck":
                # 순차 처리 병목을 병렬 처리로 변경
                adjustments.append({
                    "type": "parallelization",
                    "steps": issue["bottleneck_steps"],
                    "reason": "bottleneck_resolution"
                })
        
        return adjustments
```

### 실패 복구 전략
```python
class FailureRecoveryStrategy:
    """실패 복구 전략"""
    
    async def handle_step_failure(self, failed_step: SemanticWorkflowStep, 
                                error: Exception) -> List[SemanticWorkflowStep]:
        """단계 실패 처리"""
        
        recovery_strategies = []
        
        # 1. 대안 에이전트 시도
        if failed_step.alternative_agents:
            for alt_agent in failed_step.alternative_agents:
                recovery_step = self._create_alternative_step(failed_step, alt_agent)
                recovery_strategies.append(recovery_step)
        
        # 2. 단계 분해
        if self._can_decompose_step(failed_step):
            decomposed_steps = self._decompose_step(failed_step)
            recovery_strategies.extend(decomposed_steps)
        
        # 3. 우회 경로
        bypass_steps = self._find_bypass_path(failed_step)
        if bypass_steps:
            recovery_strategies.extend(bypass_steps)
        
        return recovery_strategies
    
    def _create_alternative_step(self, original_step: SemanticWorkflowStep, 
                               alt_agent: str) -> SemanticWorkflowStep:
        """대안 에이전트를 사용한 단계 생성"""
        return SemanticWorkflowStep(
            step_id=f"{original_step.step_id}_alt_{alt_agent}",
            semantic_purpose=original_step.semantic_purpose,
            required_concepts=original_step.required_concepts,
            agent_id=alt_agent,
            estimated_complexity=original_step.estimated_complexity,
            depends_on=original_step.depends_on,
            confidence=original_step.confidence * 0.8  # 대안이므로 신뢰도 약간 감소
        )
```

## 📊 성능 최적화

### 병렬 처리 최적화
```python
class ParallelOptimizer:
    """병렬 처리 최적화기"""
    
    def analyze_parallelization_potential(self, steps: List[SemanticWorkflowStep]) -> Dict:
        """병렬 처리 가능성 분석"""
        
        analysis = {
            "independent_groups": [],
            "dependency_chains": [],
            "parallel_potential": 0.0,
            "bottlenecks": []
        }
        
        # 의존성 그래프 구축
        dependency_graph = self._build_dependency_graph(steps)
        
        # 독립적인 그룹 식별
        independent_groups = self._find_independent_groups(dependency_graph)
        analysis["independent_groups"] = independent_groups
        
        # 병렬 처리 가능성 점수 계산
        total_steps = len(steps)
        parallelizable_steps = sum(len(group) for group in independent_groups if len(group) > 1)
        analysis["parallel_potential"] = parallelizable_steps / total_steps if total_steps > 0 else 0.0
        
        # 병목 지점 식별
        bottlenecks = self._identify_bottlenecks(dependency_graph)
        analysis["bottlenecks"] = bottlenecks
        
        return analysis
    
    def optimize_execution_order(self, steps: List[SemanticWorkflowStep]) -> List[List[SemanticWorkflowStep]]:
        """실행 순서 최적화"""
        
        # 위상 정렬을 통한 실행 순서 결정
        sorted_steps = self._topological_sort(steps)
        
        # 병렬 실행 그룹 생성
        execution_groups = []
        current_group = []
        
        for step in sorted_steps:
            if self._can_execute_in_parallel(step, current_group):
                current_group.append(step)
            else:
                if current_group:
                    execution_groups.append(current_group)
                current_group = [step]
        
        if current_group:
            execution_groups.append(current_group)
        
        return execution_groups
```

### 자원 효율성 최적화
```python
class ResourceOptimizer:
    """자원 효율성 최적화기"""
    
    def optimize_resource_usage(self, workflow_plan: OntologyWorkflowPlan) -> OntologyWorkflowPlan:
        """자원 사용량 최적화"""
        
        # 에이전트 사용량 분석
        agent_usage = self._analyze_agent_usage(workflow_plan.steps)
        
        # 중복 호출 제거
        deduplicated_steps = self._remove_duplicate_calls(workflow_plan.steps)
        
        # 캐시 활용 최적화
        cache_optimized_steps = self._optimize_cache_usage(deduplicated_steps)
        
        # 메모리 사용량 최적화
        memory_optimized_steps = self._optimize_memory_usage(cache_optimized_steps)
        
        # 최적화된 워크플로우 계획 생성
        optimized_plan = OntologyWorkflowPlan(
            plan_id=f"{workflow_plan.plan_id}_optimized",
            semantic_query=workflow_plan.semantic_query,
            steps=memory_optimized_steps,
            execution_graph=self._rebuild_execution_graph(memory_optimized_steps),
            optimization_strategy=workflow_plan.optimization_strategy,
            estimated_quality=workflow_plan.estimated_quality,
            estimated_time=self._recalculate_estimated_time(memory_optimized_steps),
            reasoning_chain=workflow_plan.reasoning_chain + ["resource_optimization_applied"]
        )
        
        return optimized_plan
```

## 📈 학습 및 개선

### 패턴 학습
```python
class WorkflowPatternLearner:
    """워크플로우 패턴 학습기"""
    
    def __init__(self):
        self.execution_history: List[Dict] = []
        self.pattern_library: Dict[str, Dict] = {}
        self.success_patterns: Dict[str, float] = {}
    
    def learn_from_execution(self, workflow_plan: OntologyWorkflowPlan, 
                           execution_results: List[AgentExecutionResult]):
        """실행 결과로부터 학습"""
        
        # 실행 기록 저장
        execution_record = {
            "query_type": workflow_plan.semantic_query.query_type,
            "complexity": workflow_plan.semantic_query.complexity,
            "agents_used": [step.agent_id for step in workflow_plan.steps],
            "execution_strategy": workflow_plan.optimization_strategy.value,
            "success_rate": sum(1 for r in execution_results if r.success) / len(execution_results),
            "total_time": sum(r.execution_time for r in execution_results),
            "timestamp": datetime.now()
        }
        
        self.execution_history.append(execution_record)
        
        # 패턴 추출 및 업데이트
        pattern_key = self._generate_pattern_key(execution_record)
        if pattern_key not in self.pattern_library:
            self.pattern_library[pattern_key] = {
                "pattern": self._extract_pattern(execution_record),
                "usage_count": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0
            }
        
        # 패턴 통계 업데이트
        pattern = self.pattern_library[pattern_key]
        pattern["usage_count"] += 1
        pattern["success_rate"] = (
            (pattern["success_rate"] * (pattern["usage_count"] - 1) + execution_record["success_rate"]) 
            / pattern["usage_count"]
        )
        pattern["avg_execution_time"] = (
            (pattern["avg_execution_time"] * (pattern["usage_count"] - 1) + execution_record["total_time"]) 
            / pattern["usage_count"]
        )
    
    def recommend_workflow_improvements(self, workflow_plan: OntologyWorkflowPlan) -> List[Dict]:
        """워크플로우 개선 제안"""
        
        recommendations = []
        
        # 유사한 성공 패턴 찾기
        similar_patterns = self._find_similar_successful_patterns(workflow_plan)
        
        for pattern in similar_patterns:
            if pattern["success_rate"] > 0.9:  # 높은 성공률 패턴
                recommendations.append({
                    "type": "pattern_adoption",
                    "description": f"성공률 {pattern['success_rate']:.1%}의 검증된 패턴 적용",
                    "suggested_changes": pattern["pattern"],
                    "expected_improvement": pattern["success_rate"] - 0.8  # 기본 성공률 대비
                })
        
        # 성능 개선 제안
        performance_improvements = self._suggest_performance_improvements(workflow_plan)
        recommendations.extend(performance_improvements)
        
        return recommendations
```

이러한 워크플로우 설계 시스템을 통해 LogosAI는 사용자 쿼리에 대해 최적화된, 지능적이고 적응적인 멀티에이전트 워크플로우를 제공할 수 있습니다. 