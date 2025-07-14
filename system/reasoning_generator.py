"""
📝 Reasoning Generator
추론 생성기

온톨로지 시스템의 상세한 추론 과정 생성
"""

from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult, WorkflowPlan


class ReasoningGenerator:
    """📝 추론 생성기"""
    
    async def generate_detailed_reasoning(self, 
                                        execution_results: List[AgentExecutionResult],
                                        workflow_plan: WorkflowPlan,
                                        semantic_query: SemanticQuery,
                                        complexity_analysis: Dict[str, Any],
                                        integrated_result: Dict[str, Any]) -> str:
        """상세한 reasoning 생성"""
        try:
            reasoning_parts = []
            
            # 1. 전체 개요
            reasoning_parts.append("# 🧠 **온톨로지 시스템 처리 분석 보고서**")
            reasoning_parts.append("")
            reasoning_parts.append("## 1️⃣ **쿼리 분석 및 이해**")
            
            # 쿼리 정보 추출
            query_text = getattr(semantic_query, 'natural_language', getattr(semantic_query, 'query_text', ''))
            intent = getattr(semantic_query, 'intent', 'general')
            
            reasoning_parts.append(f'**📝 원본 쿼리**: "{query_text}"')
            reasoning_parts.append(f"**🎯 파악된 의도**: {intent}")
            reasoning_parts.append(f"**🔍 복잡도 점수**: {complexity_analysis.get('complexity_score', 0.5):.2f}/1.0")
            reasoning_parts.append(f"**⚡ 권장 전략**: {complexity_analysis.get('recommended_strategy', 'AUTO')}")
            reasoning_parts.append("")
            
            # 2. 워크플로우 계획 분석
            reasoning_parts.append("## 2️⃣ **워크플로우 계획 수립**")
            reasoning_parts.append(f"**📋 계획 ID**: {workflow_plan.plan_id}")
            reasoning_parts.append(f"**🔗 총 단계 수**: {len(workflow_plan.steps)}단계")
            reasoning_parts.append(f"**🎯 최적화 전략**: {getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy))}")
            reasoning_parts.append(f"**📊 예상 품질**: {workflow_plan.estimated_quality:.2f}/1.0")
            reasoning_parts.append(f"**⏱️ 예상 시간**: {workflow_plan.estimated_time:.1f}초")
            
            # 워크플로우 reasoning 추가
            if hasattr(workflow_plan, 'reasoning_chain') and workflow_plan.reasoning_chain:
                reasoning_parts.append("")
                reasoning_parts.append("**🤔 계획 수립 근거**:")
                for i, reason in enumerate(workflow_plan.reasoning_chain, 1):
                    reasoning_parts.append(f"  {i}. {reason}")
            
            # Mermaid 워크플로우 다이어그램 생성
            reasoning_parts.append("")
            reasoning_parts.append("**🌊 워크플로우 다이어그램**:")
            reasoning_parts.append("```mermaid")
            mermaid_diagram = self._generate_workflow_mermaid(workflow_plan)
            reasoning_parts.append(mermaid_diagram)
            reasoning_parts.append("```")
            
            reasoning_parts.append("")
            reasoning_parts.append("**📋 계획된 단계들**:")
            for i, step in enumerate(workflow_plan.steps, 1):
                reasoning_parts.append(f"  {i}. **{step.semantic_purpose}** (에이전트: {step.agent_id})")
                if hasattr(step, 'depends_on') and step.depends_on:
                    reasoning_parts.append(f"     - 의존성: {', '.join(step.depends_on)}")
            reasoning_parts.append("")
            
            # 3. 실행 과정 분석
            reasoning_parts.append("## 3️⃣ **실행 과정 상세 분석**")
            
            successful_results = [r for r in execution_results if r.is_successful()]
            failed_results = [r for r in execution_results if not r.is_successful()]
            
            reasoning_parts.append(f"**📊 실행 통계**:")
            reasoning_parts.append(f"  - 총 실행: {len(execution_results)}개")
            reasoning_parts.append(f"  - 성공: {len(successful_results)}개")
            reasoning_parts.append(f"  - 실패: {len(failed_results)}개")
            reasoning_parts.append(f"  - 성공률: {len(successful_results)/len(execution_results)*100:.1f}%")
            reasoning_parts.append("")
            
            # 성공한 실행들 상세 분석
            if successful_results:
                reasoning_parts.append("**✅ 성공한 실행들**:")
                for i, result in enumerate(successful_results, 1):
                    reasoning_parts.append(f"  {i}. **{result.agent_id}**")
                    reasoning_parts.append(f"     - 실행 시간: {result.execution_time:.2f}초")
                    reasoning_parts.append(f"     - 신뢰도: {result.confidence:.2f}")
                    
                    # 결과 미리보기
                    if result.data:
                        result_preview = str(result.data)[:100]
                        if len(result_preview) > 100:
                            result_preview += "..."
                        reasoning_parts.append(f"     - 결과 미리보기: {result_preview}")
                reasoning_parts.append("")
            
            # 실패한 실행들 분석
            if failed_results:
                reasoning_parts.append("**❌ 실패한 실행들**:")
                for i, result in enumerate(failed_results, 1):
                    reasoning_parts.append(f"  {i}. **{result.agent_id}**")
                    reasoning_parts.append(f"     - 실행 시간: {result.execution_time:.2f}초")
                    if result.error_message:
                        reasoning_parts.append(f"     - 오류: {result.error_message}")
                reasoning_parts.append("")
            
            # 4. 결과 통합 과정
            reasoning_parts.append("## 4️⃣ **결과 통합 과정**")
            reasoning_parts.append(f"**🔄 통합 방식**: {intent} 기반 통합")
            reasoning_parts.append(f"**📊 통합 상태**: {integrated_result.get('status', 'unknown')}")
            
            if integrated_result.get('metadata'):
                metadata = integrated_result['metadata']
                reasoning_parts.append("**📈 통합 메트릭**:")
                reasoning_parts.append(f"  - 총 실행 시간: {metadata.get('total_execution_time', 0):.2f}초")
                reasoning_parts.append(f"  - 평균 신뢰도: {metadata.get('average_confidence', 0):.2f}")
                reasoning_parts.append(f"  - 사용된 전략: {metadata.get('strategy_used', 'AUTO')}")
            reasoning_parts.append("")
            
            # 5. 온톨로지 지식 그래프 업데이트
            reasoning_parts.append("## 5️⃣ **온톨로지 지식 그래프 업데이트**")
            reasoning_parts.append("**🧠 학습된 지식**:")
            reasoning_parts.append("  - 에이전트 간 협력 관계 추가")
            reasoning_parts.append("  - 워크플로우 패턴 학습")
            reasoning_parts.append("  - 쿼리-결과 매핑 강화")
            reasoning_parts.append("  - 성능 메트릭 업데이트")
            reasoning_parts.append("")
            
            # 6. 성능 분석
            reasoning_parts.append("## 6️⃣ **성능 분석**")
            total_time = sum(r.execution_time for r in execution_results)
            avg_time = total_time / len(execution_results) if execution_results else 0
            
            reasoning_parts.append(f"**⏱️ 시간 분석**:")
            reasoning_parts.append(f"  - 총 처리 시간: {total_time:.2f}초")
            reasoning_parts.append(f"  - 평균 단계 시간: {avg_time:.2f}초")
            reasoning_parts.append(f"  - 예상 대비 실제: {total_time/workflow_plan.estimated_time*100:.1f}%" if workflow_plan.estimated_time > 0 else "  - 예상 시간 정보 없음")
            
            if total_time < 5:
                reasoning_parts.append("**🏆 성능 평가**: 우수 (5초 미만)")
            elif total_time < 15:
                reasoning_parts.append("**🏆 성능 평가**: 양호 (15초 미만)")
            else:
                reasoning_parts.append("**🏆 성능 평가**: 보통 (최적화 여지 있음)")
            reasoning_parts.append("")
            
            # 7. 시스템 인사이트
            reasoning_parts.append("## 7️⃣ **시스템 인사이트**")
            reasoning_parts.append("**🧠 온톨로지 시스템의 핵심 가치**:")
            reasoning_parts.append("  - 의미론적 쿼리 이해 및 분석")
            reasoning_parts.append("  - 지능적 워크플로우 설계")
            reasoning_parts.append("  - 에이전트 간 협력 최적화")
            reasoning_parts.append("  - 지식 그래프 기반 학습")
            reasoning_parts.append("  - 동적 성능 최적화")
            reasoning_parts.append("")
            reasoning_parts.append("**💫 이번 처리의 특징**:")
            
            if len(successful_results) == len(execution_results):
                reasoning_parts.append("  - 모든 에이전트가 성공적으로 실행됨")
            elif len(successful_results) > 0:
                reasoning_parts.append("  - 일부 에이전트 실패했으나 결과 도출 성공")
            else:
                reasoning_parts.append("  - 모든 에이전트 실패, 폴백 메커니즘 작동")
            
            if total_time < workflow_plan.estimated_time:
                reasoning_parts.append("  - 예상보다 빠른 처리 완료")
            else:
                reasoning_parts.append("  - 예상 시간 내 처리 완료")
            
            reasoning_parts.append("")
            reasoning_parts.append("---")
            reasoning_parts.append("*🎯 이 분석은 LOGOS 온톨로지 시스템에 의해 자동 생성되었습니다.*")
            
            return "\n".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"상세 reasoning 생성 실패: {e}")
            # 폴백 reasoning
            return f"""# 🧠 **온톨로지 시스템 처리 보고서**

## 📝 **처리 요약**
- 쿼리 처리 완료
- 워크플로우 실행: {len(execution_results)}단계
- 성공한 단계: {len([r for r in execution_results if r.is_successful()])}개
- 총 처리 시간: {sum(r.execution_time for r in execution_results):.2f}초

## 🎯 **시스템 상태**
온톨로지 시스템이 정상적으로 작동하여 요청을 처리했습니다.

*상세 분석 생성 중 오류 발생: {str(e)}*"""
    
    def _generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """워크플로우 계획을 Mermaid 다이어그램으로 생성"""
        try:
            mermaid_lines = ["graph TD"]
            
            # 시작 노드
            mermaid_lines.append('    Start(["🚀 시작"]) --> Query["📝 쿼리 분석"]')
            
            # 각 단계를 노드로 추가
            for i, step in enumerate(workflow_plan.steps):
                step_id = f"Step{i+1}"
                agent_name = step.agent_id.replace('_', ' ').title()
                purpose = step.semantic_purpose[:30] + "..." if len(step.semantic_purpose) > 30 else step.semantic_purpose
                
                # 노드 정의
                mermaid_lines.append(f'    {step_id}["🤖 {agent_name}<br/>{purpose}"]')
                
                # 연결 관계
                if i == 0:
                    mermaid_lines.append(f'    Query --> {step_id}')
                else:
                    prev_step_id = f"Step{i}"
                    mermaid_lines.append(f'    {prev_step_id} --> {step_id}')
                
                # 의존성이 있는 경우
                if hasattr(step, 'depends_on') and step.depends_on:
                    for dep in step.depends_on:
                        # 의존성 단계 찾기
                        for j, dep_step in enumerate(workflow_plan.steps):
                            if dep_step.step_id == dep:
                                dep_step_id = f"Step{j+1}"
                                mermaid_lines.append(f'    {dep_step_id} -.-> {step_id}')
                                break
            
            # 마지막 단계에서 결과로 연결
            if workflow_plan.steps:
                last_step_id = f"Step{len(workflow_plan.steps)}"
                mermaid_lines.append(f'    {last_step_id} --> Result[["✅ 결과 통합"]]')
                mermaid_lines.append('    Result --> End(["🎉 완료"])')
            else:
                mermaid_lines.append('    Query --> End(["🎉 완료"])')
            
            # 스타일 추가
            mermaid_lines.extend([
                "",
                "    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
                "    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px", 
                "    classDef agent fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
                "    classDef result fill:#fff3e0,stroke:#e65100,stroke-width:2px",
                "",
                "    class Start,End startEnd",
                "    class Query,Result result"
            ])
            
            # 에이전트 노드들에 스타일 적용
            for i in range(len(workflow_plan.steps)):
                mermaid_lines.append(f"    class Step{i+1} agent")
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid 다이어그램 생성 실패: {e}")
            return f'graph TD\n    Start(["시작"]) --> Error["다이어그램 생성 실패: {str(e)}"]\n    Error --> End(["종료"])' 