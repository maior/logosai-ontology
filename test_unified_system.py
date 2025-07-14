#!/usr/bin/env python3
"""
🚀 통합 쿼리 처리 시스템 테스트
Unified Query Processing System Test

실제 agents.json 파일의 에이전트 정보를 활용하여
통합 LLM 처리 시스템의 효율성을 테스트합니다.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

# 테스트 대상 모듈 import (예외 처리로 안전하게)
try:
    from core.unified_query_processor import get_unified_query_processor
    UNIFIED_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 통합 프로세서 import 실패: {e}")
    UNIFIED_PROCESSOR_AVAILABLE = False

try:
    from core.enhanced_query_processor import get_enhanced_query_processor
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 향상된 프로세서 import 실패: {e}")
    ENHANCED_PROCESSOR_AVAILABLE = False


class UnifiedSystemTester:
    """🧪 통합 시스템 테스터"""
    
    def __init__(self):
        #self.test_query = "오늘 날씨를 확인하고, 환율 정보도 알려주고, 삼성전자 주가 정보도 알려줘."
        #self.test_query = "오늘 금 시세 확인하고 1온스당 원화 가격 계산해서 정리해줘."
        self.test_query = "AI 기술 트렌드를 조사해서 분석하고 시각적 보고서로 만들어줘"
        self.agents_json_path = "/Users/maior/Development/skku/Logos/logosai/logosai/examples/configs/agents.json"
        self.test_agents_info = []
        self.available_agents = []
        
    def load_agents_info(self):
        """agents.json에서 실제 에이전트 정보 로드"""
        try:
            if Path(self.agents_json_path).exists():
                with open(self.agents_json_path, 'r', encoding='utf-8') as f:
                    agents_data = json.load(f)
                
                # agents 배열에서 에이전트 정보 추출
                for agent in agents_data.get('agents', []):
                    agent_id = agent.get('agent_id')
                    if agent_id:
                        self.test_agents_info.append({
                            'agent_id': agent_id,
                            'agent_data': agent
                        })
                        self.available_agents.append(agent_id)
                
                print(f"✅ agents.json에서 {len(self.test_agents_info)}개 에이전트 로드 완료")
                
                # 로드된 에이전트 목록 출력
                for agent in self.test_agents_info[:5]:  # 처음 5개만 출력
                    agent_data = agent['agent_data']
                    agent_name = agent_data.get('name', agent['agent_id'])
                    agent_type = agent_data.get('metadata', {}).get('agent_type', 'UNKNOWN')
                    print(f"  📋 {agent['agent_id']}: {agent_name} ({agent_type})")
                
                if len(self.test_agents_info) > 5:
                    print(f"  ... 및 {len(self.test_agents_info) - 5}개 더")
                    
            else:
                print(f"❌ agents.json 파일을 찾을 수 없음: {self.agents_json_path}")
                self._create_fallback_agents()
                
        except Exception as e:
            print(f"❌ agents.json 로드 실패: {e}")
            self._create_fallback_agents()
    
    def _create_fallback_agents(self):
        """폴백용 에이전트 정보 생성"""
        fallback_agents = [
            {
                'agent_id': 'llm_search_agent',
                'agent_data': {
                    'name': 'LLM 검색 에이전트',
                    'description': '일반적인 검색 및 정보 제공 에이전트',
                    'metadata': {'agent_type': 'LLM_SEARCH'},
                    'capabilities': [
                        {'name': '지식 검색', 'description': 'LLM 지식 기반 검색'},
                        {'name': '정보 제공', 'description': '다양한 주제의 정보 제공'}
                    ],
                    'examples': ['날씨 정보 검색', '주식 정보 조회', '환율 정보'],
                    'tags': ['검색', '정보제공', 'LLM']
                }
            },
            {
                'agent_id': 'analysis_agent', 
                'agent_data': {
                    'name': '데이터 분석 에이전트',
                    'description': '데이터 분석 전문 에이전트',
                    'metadata': {'agent_type': 'ANALYSIS'},
                    'capabilities': [
                        {'name': '통계 분석', 'description': '기본 통계량 분석'},
                        {'name': '데이터 분석', 'description': '다양한 데이터 분석'}
                    ],
                    'examples': ['데이터 통계 분석', '트렌드 분석'],
                    'tags': ['분석', '데이터', '통계']
                }
            },
            {
                'agent_id': 'crawler_agent',
                'agent_data': {
                    'name': '크롤링 에이전트',
                    'description': '웹 크롤링 전문 에이전트',
                    'metadata': {'agent_type': 'CRAWLER'},
                    'capabilities': [
                        {'name': '웹 크롤링', 'description': '웹사이트 데이터 수집'},
                        {'name': '데이터 추출', 'description': '웹 데이터 추출'}
                    ],
                    'examples': ['웹사이트 크롤링', '데이터 수집'],
                    'tags': ['크롤링', '웹스크래핑', '데이터수집']
                }
            }
        ]
        
        self.test_agents_info = fallback_agents
        self.available_agents = [agent['agent_id'] for agent in fallback_agents]
        print(f"🔄 폴백 에이전트 {len(fallback_agents)}개 생성 완료")

    async def test_unified_processor(self):
        """통합 프로세서 테스트"""
        print("\n" + "="*60)
        print("🚀 통합 쿼리 프로세서 테스트")
        print("="*60)
        
        if not UNIFIED_PROCESSOR_AVAILABLE:
            print("❌ 통합 프로세서를 사용할 수 없습니다.")
            return None
            
        try:
            processor = get_unified_query_processor()
            processor.set_installed_agents_info(self.test_agents_info)
            
            print(f"\n📝 테스트 쿼리: {self.test_query}")
            print(f"🤖 사용 가능한 에이전트: {len(self.available_agents)}개")
            
            start_time = datetime.now()
            
            # 통합 처리 실행
            result = await processor.process_unified_query(
                self.test_query, 
                self.available_agents #[:5]  # 처음 5개 에이전트만 사용
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"\n⏱️ 처리 시간: {processing_time:.2f}초")
            print(f"🎯 처리 결과:")
            
            # 결과 분석
            query_analysis = result.get('query_analysis', {})
            print(f"  📊 쿼리 분석:")
            print(f"    - 복잡도: {query_analysis.get('complexity', 'unknown')}")
            print(f"    - 멀티 작업: {query_analysis.get('multi_task', False)}")
            print(f"    - 작업 수: {query_analysis.get('task_count', 0)}")
            print(f"    - 주요 의도: {query_analysis.get('primary_intent', 'unknown')}")
            
            # 작업 분해 결과
            task_breakdown = result.get('task_breakdown', [])
            print(f"\n  🔍 작업 분해 결과 ({len(task_breakdown)}개):")
            for i, task in enumerate(task_breakdown, 1):
                print(f"    {i}. {task.get('task_description', 'No description')}")
                print(f"       도메인: {task.get('domain', 'unknown')}")
                print(f"       키워드: {task.get('extracted_keywords', [])}")
            
            # 에이전트 매핑 결과
            agent_mappings = result.get('agent_mappings', [])
            print(f"\n  🎯 에이전트 매핑 결과 ({len(agent_mappings)}개):")
            for i, mapping in enumerate(agent_mappings, 1):
                # confidence 값 안전하게 처리
                confidence = mapping.get('confidence', 0)
                try:
                    confidence_float = float(confidence)
                except (ValueError, TypeError):
                    confidence_float = 0.0
                
                print(f"    {i}. 에이전트: {mapping.get('selected_agent', 'unknown')}")
                print(f"       타입: {mapping.get('agent_type', 'unknown')}")
                optimized_msg = mapping.get('individual_query', mapping.get('optimized_message', 'No message'))
                print(f"       최적화된 메시지: {optimized_msg[:80]}...")
                print(f"       신뢰도: {confidence_float:.2f}")
            
            # 실행 계획
            execution_plan = result.get('execution_plan', {})
            # 예상 시간도 안전하게 처리
            estimated_time = execution_plan.get('estimated_time', 0)
            try:
                estimated_time_float = float(estimated_time)
            except (ValueError, TypeError):
                estimated_time_float = 0.0
                
            print(f"\n  ⚡ 실행 계획:")
            print(f"    - 전략: {execution_plan.get('strategy', 'unknown')}")
            print(f"    - 예상 시간: {estimated_time_float:.1f}초")
            print(f"    - 병렬 그룹: {execution_plan.get('parallel_groups', [])}")
            
            # 품질 평가
            quality = result.get('quality_assessment', {})
            print(f"\n  📈 품질 평가:")
            
            # 각 품질 지표도 안전하게 처리
            completeness = quality.get('completeness', 0)
            try:
                completeness_float = float(completeness)
            except (ValueError, TypeError):
                completeness_float = 0.0
                
            agent_match_quality = quality.get('agent_match_quality', 0)
            try:
                agent_match_quality_float = float(agent_match_quality)
            except (ValueError, TypeError):
                agent_match_quality_float = 0.0
                
            execution_efficiency = quality.get('execution_efficiency', 0)
            try:
                execution_efficiency_float = float(execution_efficiency)
            except (ValueError, TypeError):
                execution_efficiency_float = 0.0
                
            overall_confidence = quality.get('overall_confidence', 0)
            try:
                overall_confidence_float = float(overall_confidence)
            except (ValueError, TypeError):
                overall_confidence_float = 0.0
            
            print(f"    - 완성도: {completeness_float:.2f}")
            print(f"    - 에이전트 매칭 품질: {agent_match_quality_float:.2f}")
            print(f"    - 실행 효율성: {execution_efficiency_float:.2f}")
            print(f"    - 전체 신뢰도: {overall_confidence_float:.2f}")
            
            if result.get('fallback_mode'):
                print(f"\n⚠️ 폴백 모드로 처리됨")
            
            return result
            
        except Exception as e:
            print(f"❌ 통합 프로세서 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_enhanced_processor(self):
        """기존 향상된 프로세서 테스트 (비교용)"""
        print("\n" + "="*60)
        print("🔧 기존 향상된 쿼리 프로세서 테스트 (비교용)")
        print("="*60)
        
        if not ENHANCED_PROCESSOR_AVAILABLE:
            print("❌ 향상된 프로세서를 사용할 수 없습니다.")
            return None
            
        try:
            processor = get_enhanced_query_processor()
            processor.set_installed_agents_info(self.test_agents_info)
            
            print(f"\n📝 테스트 쿼리: {self.test_query}")
            
            start_time = datetime.now()
            
            # 기존 처리 실행
            decomposition, mappings = await processor.process_complex_query(
                self.test_query, 
                self.available_agents[:5]
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"\n⏱️ 처리 시간: {processing_time:.2f}초")
            print(f"🎯 처리 결과:")
            print(f"  - 실행 전략: {decomposition.execution_strategy}")
            print(f"  - 쿼리 부분 수: {len(decomposition.query_parts)}")
            print(f"  - 에이전트 매핑 수: {len(mappings)}")
            
            print(f"\n  🔍 에이전트 매핑:")
            for i, mapping in enumerate(mappings, 1):
                print(f"    {i}. {mapping.agent_id}: {mapping.optimized_query[:60]}...")
            
            return {"decomposition": decomposition, "mappings": mappings}
            
        except Exception as e:
            print(f"❌ 기존 프로세서 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_results(self, unified_result, enhanced_result):
        """결과 비교 분석"""
        print("\n" + "="*60)
        print("📊 성능 비교 분석")
        print("="*60)
        
        if not unified_result or not enhanced_result:
            print("❌ 비교할 결과가 충분하지 않습니다.")
            return
        
        print("🚀 통합 시스템 vs 🔧 기존 시스템:")
        
        # 에이전트 수 비교
        unified_agents = len(unified_result.get('agent_mappings', []))
        enhanced_agents = len(enhanced_result.get('mappings', []))
        print(f"  📊 선택된 에이전트 수: {unified_agents} vs {enhanced_agents}")
        
        # LLM 호출 횟수 (추정)
        print(f"  🧠 추정 LLM 호출 횟수: 1회 vs 2-3회")
        
        # 결과 복잡도
        unified_complexity = unified_result.get('query_analysis', {}).get('complexity', 'unknown')
        print(f"  🔍 분석 복잡도: {unified_complexity}")
        
        # 품질 점수
        quality = unified_result.get('quality_assessment', {})
        if quality:
            print(f"  📈 품질 점수:")
            
            # 안전한 포맷팅
            completeness = quality.get('completeness', 0)
            try:
                completeness_float = float(completeness)
            except (ValueError, TypeError):
                completeness_float = 0.0
                
            agent_match_quality = quality.get('agent_match_quality', 0)
            try:
                agent_match_quality_float = float(agent_match_quality)
            except (ValueError, TypeError):
                agent_match_quality_float = 0.0
                
            execution_efficiency = quality.get('execution_efficiency', 0)
            try:
                execution_efficiency_float = float(execution_efficiency)
            except (ValueError, TypeError):
                execution_efficiency_float = 0.0
            
            print(f"    - 완성도: {completeness_float:.2f}")
            print(f"    - 에이전트 매칭: {agent_match_quality_float:.2f}")
            print(f"    - 실행 효율성: {execution_efficiency_float:.2f}")

    async def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🧪 통합 쿼리 처리 시스템 종합 테스트")
        print("="*60)
        
        # 에이전트 정보 로드
        self.load_agents_info()
        
        if not self.test_agents_info:
            print("❌ 테스트할 에이전트 정보가 없습니다.")
            return
        
        # 통합 프로세서 테스트
        unified_result = await self.test_unified_processor()
        
        # 기존 프로세서 테스트 (비교용)
        enhanced_result = await self.test_enhanced_processor()
        
        # 결과 비교
        self.compare_results(unified_result, enhanced_result)
        
        print("\n🎯 테스트 완료!")
        print("\n💡 주요 개선사항:")
        print("  1. LLM 호출 횟수 감소: 3회 → 1회")
        print("  2. 실제 에이전트 정보 활용")
        print("  3. 통합된 분석 및 최적화")
        print("  4. 더 정확한 에이전트 매칭")

        # 정규식 기반 분석 테스트
        print("\n============================================================")
        print("🔍 정규식 기반 분석 테스트 (폴백 모드)")
        print("============================================================")
        
        if UNIFIED_PROCESSOR_AVAILABLE:
            try:
                # 직접 정규식 분석 테스트  
                unified_proc = get_unified_query_processor()
                unified_proc.set_installed_agents_info(self.test_agents_info)
                result = unified_proc._create_emergency_fallback(
                    self.test_query, self.available_agents, 'ko'
                )
                
                print(f"⏱️ 처리 시간: 즉시")
                print(f"🎯 처리 결과:")
                print(f"  📊 쿼리 분석:")
                print(f"    - 복잡도: {result.get('query_analysis', {}).get('complexity', 'unknown')}")
                print(f"    - 멀티 작업: {result.get('query_analysis', {}).get('multi_task', False)}")
                print(f"    - 작업 수: {result.get('query_analysis', {}).get('task_count', 0)}")
                print(f"    - 도메인: {result.get('query_analysis', {}).get('domains', [])}")
                
                task_breakdown = result.get('task_breakdown', [])
                print(f"\n  🔍 작업 분해 결과 ({len(task_breakdown)}개):")
                for i, task in enumerate(task_breakdown, 1):
                    print(f"    {i}. {task.get('task_description', 'Unknown')}")
                    print(f"       도메인: {task.get('domain', 'unknown')}")
                    print(f"       키워드: {task.get('extracted_keywords', [])}")
                
                agent_mappings = result.get('agent_mappings', [])
                print(f"\n  🎯 에이전트 매핑 결과 ({len(agent_mappings)}개):")
                for i, mapping in enumerate(agent_mappings, 1):
                    # confidence 값 안전하게 처리
                    confidence = mapping.get('confidence', 0.0)
                    try:
                        confidence_float = float(confidence)
                    except (ValueError, TypeError):
                        confidence_float = 0.0
                    
                    print(f"    {i}. 에이전트: {mapping.get('selected_agent', 'unknown')}")
                    print(f"       타입: {mapping.get('agent_type', 'unknown')}")
                    optimized_msg = mapping.get('individual_query', mapping.get('optimized_message', ''))
                    print(f"       최적화된 메시지: {optimized_msg[:80]}...")
                    print(f"       신뢰도: {confidence_float:.2f}")
                
                execution_plan = result.get('execution_plan', {})
                print(f"\n  ⚡ 실행 계획:")
                print(f"    - 전략: {execution_plan.get('strategy', 'unknown')}")
                print(f"    - 예상 시간: {execution_plan.get('estimated_time', 0)}초")
                print(f"    - 병렬 그룹: {execution_plan.get('parallel_groups', [])}")
                
                quality_assessment = result.get('quality_assessment', {})
                print(f"\n  📈 품질 평가:")
                
                # 각 품질 지표도 안전하게 처리
                completeness = quality_assessment.get('completeness', 0.0)
                try:
                    completeness_float = float(completeness)
                except (ValueError, TypeError):
                    completeness_float = 0.0
                    
                agent_match_quality = quality_assessment.get('agent_match_quality', 0.0)
                try:
                    agent_match_quality_float = float(agent_match_quality)
                except (ValueError, TypeError):
                    agent_match_quality_float = 0.0
                    
                execution_efficiency = quality_assessment.get('execution_efficiency', 0.0)
                try:
                    execution_efficiency_float = float(execution_efficiency)
                except (ValueError, TypeError):
                    execution_efficiency_float = 0.0
                    
                overall_confidence = quality_assessment.get('overall_confidence', 0.0)
                try:
                    overall_confidence_float = float(overall_confidence)
                except (ValueError, TypeError):
                    overall_confidence_float = 0.0
                
                print(f"    - 완성도: {completeness_float:.2f}")
                print(f"    - 에이전트 매칭 품질: {agent_match_quality_float:.2f}")
                print(f"    - 실행 효율성: {execution_efficiency_float:.2f}")
                print(f"    - 전체 신뢰도: {overall_confidence_float:.2f}")
                
            except Exception as e:
                print(f"❌ 정규식 분석 테스트 실패: {e}")
        else:
            print("❌ 통합 프로세서를 사용할 수 없습니다.")


async def main():
    """메인 테스트 함수"""
    tester = UnifiedSystemTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main()) 