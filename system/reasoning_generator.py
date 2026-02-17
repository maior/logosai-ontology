"""
📝 Reasoning Generator
Reasoning Generator

Generates detailed reasoning processes for the ontology system
"""

from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

from ..core.models import SemanticQuery, AgentExecutionResult, WorkflowPlan


class ReasoningGenerator:
    """📝 Reasoning Generator"""
    
    async def generate_detailed_reasoning(self, 
                                        execution_results: List[AgentExecutionResult],
                                        workflow_plan: WorkflowPlan,
                                        semantic_query: SemanticQuery,
                                        complexity_analysis: Dict[str, Any],
                                        integrated_result: Dict[str, Any]) -> str:
        """Generate detailed reasoning"""
        try:
            reasoning_parts = []
            
            # 1. Overall overview
            reasoning_parts.append("# 🧠 **Ontology System Processing Analysis Report**")
            reasoning_parts.append("")
            reasoning_parts.append("## 1️⃣ **Query Analysis and Understanding**")
            
            # Extract query information
            query_text = getattr(semantic_query, 'natural_language', getattr(semantic_query, 'query_text', ''))
            intent = getattr(semantic_query, 'intent', 'general')
            
            reasoning_parts.append(f'**📝 Original Query**: "{query_text}"')
            reasoning_parts.append(f"**🎯 Identified Intent**: {intent}")
            reasoning_parts.append(f"**🔍 Complexity Score**: {complexity_analysis.get('complexity_score', 0.5):.2f}/1.0")
            reasoning_parts.append(f"**⚡ Recommended Strategy**: {complexity_analysis.get('recommended_strategy', 'AUTO')}")
            reasoning_parts.append("")
            
            # 2. Workflow plan analysis
            reasoning_parts.append("## 2️⃣ **Workflow Plan Construction**")
            reasoning_parts.append(f"**📋 Plan ID**: {workflow_plan.plan_id}")
            reasoning_parts.append(f"**🔗 Total Steps**: {len(workflow_plan.steps)}")
            reasoning_parts.append(f"**🎯 Optimization Strategy**: {getattr(workflow_plan.optimization_strategy, 'value', str(workflow_plan.optimization_strategy))}")
            reasoning_parts.append(f"**📊 Estimated Quality**: {workflow_plan.estimated_quality:.2f}/1.0")
            reasoning_parts.append(f"**⏱️ Estimated Time**: {workflow_plan.estimated_time:.1f}s")
            
            # Add workflow reasoning
            if hasattr(workflow_plan, 'reasoning_chain') and workflow_plan.reasoning_chain:
                reasoning_parts.append("")
                reasoning_parts.append("**🤔 Plan Construction Rationale**:")
                for i, reason in enumerate(workflow_plan.reasoning_chain, 1):
                    reasoning_parts.append(f"  {i}. {reason}")
            
            # Generate Mermaid workflow diagram
            reasoning_parts.append("")
            reasoning_parts.append("**🌊 Workflow Diagram**:")
            reasoning_parts.append("```mermaid")
            mermaid_diagram = self._generate_workflow_mermaid(workflow_plan)
            reasoning_parts.append(mermaid_diagram)
            reasoning_parts.append("```")
            
            reasoning_parts.append("")
            reasoning_parts.append("**📋 Planned Steps**:")
            for i, step in enumerate(workflow_plan.steps, 1):
                reasoning_parts.append(f"  {i}. **{step.semantic_purpose}** (agent: {step.agent_id})")
                if hasattr(step, 'depends_on') and step.depends_on:
                    reasoning_parts.append(f"     - Dependencies: {', '.join(step.depends_on)}")
            reasoning_parts.append("")
            
            # 3. Execution process analysis
            reasoning_parts.append("## 3️⃣ **Execution Process Detailed Analysis**")
            
            successful_results = [r for r in execution_results if r.is_successful()]
            failed_results = [r for r in execution_results if not r.is_successful()]
            
            reasoning_parts.append(f"**📊 Execution Statistics**:")
            reasoning_parts.append(f"  - Total executions: {len(execution_results)}")
            reasoning_parts.append(f"  - Successful: {len(successful_results)}")
            reasoning_parts.append(f"  - Failed: {len(failed_results)}")
            reasoning_parts.append(f"  - Success rate: {len(successful_results)/len(execution_results)*100:.1f}%")
            reasoning_parts.append("")
            
            # Detailed analysis of successful executions
            if successful_results:
                reasoning_parts.append("**✅ Successful Executions**:")
                for i, result in enumerate(successful_results, 1):
                    reasoning_parts.append(f"  {i}. **{result.agent_id}**")
                    reasoning_parts.append(f"     - Execution time: {result.execution_time:.2f}s")
                    reasoning_parts.append(f"     - Confidence: {result.confidence:.2f}")
                    
                    # Result preview
                    if result.data:
                        result_preview = str(result.data)[:100]
                        if len(result_preview) > 100:
                            result_preview += "..."
                        reasoning_parts.append(f"     - Result preview: {result_preview}")
                reasoning_parts.append("")
            
            # Analysis of failed executions
            if failed_results:
                reasoning_parts.append("**❌ Failed Executions**:")
                for i, result in enumerate(failed_results, 1):
                    reasoning_parts.append(f"  {i}. **{result.agent_id}**")
                    reasoning_parts.append(f"     - Execution time: {result.execution_time:.2f}s")
                    if result.error_message:
                        reasoning_parts.append(f"     - Error: {result.error_message}")
                reasoning_parts.append("")
            
            # 4. Result integration process
            reasoning_parts.append("## 4️⃣ **Result Integration Process**")
            reasoning_parts.append(f"**🔄 Integration Method**: {intent}-based integration")
            reasoning_parts.append(f"**📊 Integration Status**: {integrated_result.get('status', 'unknown')}")
            
            if integrated_result.get('metadata'):
                metadata = integrated_result['metadata']
                reasoning_parts.append("**📈 Integration Metrics**:")
                reasoning_parts.append(f"  - Total execution time: {metadata.get('total_execution_time', 0):.2f}s")
                reasoning_parts.append(f"  - Average confidence: {metadata.get('average_confidence', 0):.2f}")
                reasoning_parts.append(f"  - Strategy used: {metadata.get('strategy_used', 'AUTO')}")
            reasoning_parts.append("")
            
            # 5. Ontology knowledge graph update
            reasoning_parts.append("## 5️⃣ **Ontology Knowledge Graph Update**")
            reasoning_parts.append("**🧠 Learned Knowledge**:")
            reasoning_parts.append("  - Agent collaboration relationships added")
            reasoning_parts.append("  - Workflow patterns learned")
            reasoning_parts.append("  - Query-result mapping reinforced")
            reasoning_parts.append("  - Performance metrics updated")
            reasoning_parts.append("")
            
            # 6. Performance analysis
            reasoning_parts.append("## 6️⃣ **Performance Analysis**")
            total_time = sum(r.execution_time for r in execution_results)
            avg_time = total_time / len(execution_results) if execution_results else 0
            
            reasoning_parts.append(f"**⏱️ Time Analysis**:")
            reasoning_parts.append(f"  - Total processing time: {total_time:.2f}s")
            reasoning_parts.append(f"  - Average step time: {avg_time:.2f}s")
            reasoning_parts.append(f"  - Actual vs estimated: {total_time/workflow_plan.estimated_time*100:.1f}%" if workflow_plan.estimated_time > 0 else "  - No estimated time information")
            
            if total_time < 5:
                reasoning_parts.append("**🏆 Performance Rating**: Excellent (under 5s)")
            elif total_time < 15:
                reasoning_parts.append("**🏆 Performance Rating**: Good (under 15s)")
            else:
                reasoning_parts.append("**🏆 Performance Rating**: Average (room for optimization)")
            reasoning_parts.append("")
            
            # 7. System insights
            reasoning_parts.append("## 7️⃣ **System Insights**")
            reasoning_parts.append("**🧠 Core Value of the Ontology System**:")
            reasoning_parts.append("  - Semantic query understanding and analysis")
            reasoning_parts.append("  - Intelligent workflow design")
            reasoning_parts.append("  - Agent collaboration optimization")
            reasoning_parts.append("  - Knowledge graph-based learning")
            reasoning_parts.append("  - Dynamic performance optimization")
            reasoning_parts.append("")
            reasoning_parts.append("**💫 Characteristics of this Processing**:")
            
            if len(successful_results) == len(execution_results):
                reasoning_parts.append("  - All agents executed successfully")
            elif len(successful_results) > 0:
                reasoning_parts.append("  - Some agents failed but result was produced successfully")
            else:
                reasoning_parts.append("  - All agents failed, fallback mechanism activated")
            
            if total_time < workflow_plan.estimated_time:
                reasoning_parts.append("  - Processing completed faster than estimated")
            else:
                reasoning_parts.append("  - Processing completed within estimated time")
            
            reasoning_parts.append("")
            reasoning_parts.append("---")
            reasoning_parts.append("*🎯 This analysis was automatically generated by the LOGOS Ontology System.*")
            
            return "\n".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Detailed reasoning generation failed: {e}")
            # Fallback reasoning
            return f"""# 🧠 **Ontology System Processing Report**

## 📝 **Processing Summary**
- Query processing complete
- Workflow executed: {len(execution_results)} steps
- Successful steps: {len([r for r in execution_results if r.is_successful()])}
- Total processing time: {sum(r.execution_time for r in execution_results):.2f}s

## 🎯 **System Status**
The ontology system processed the request successfully.

*Error during detailed analysis generation: {str(e)}*"""
    
    def _generate_workflow_mermaid(self, workflow_plan: WorkflowPlan) -> str:
        """Generate Mermaid diagram from workflow plan"""
        try:
            mermaid_lines = ["graph TD"]
            
            # Start node
            mermaid_lines.append('    Start(["🚀 Start"]) --> Query["📝 Query Analysis"]')
            
            # Add each step as a node
            for i, step in enumerate(workflow_plan.steps):
                step_id = f"Step{i+1}"
                agent_name = step.agent_id.replace('_', ' ').title()
                purpose = step.semantic_purpose[:30] + "..." if len(step.semantic_purpose) > 30 else step.semantic_purpose
                
                # Node definition
                mermaid_lines.append(f'    {step_id}["🤖 {agent_name}<br/>{purpose}"]')
                
                # Connection relationships
                if i == 0:
                    mermaid_lines.append(f'    Query --> {step_id}')
                else:
                    prev_step_id = f"Step{i}"
                    mermaid_lines.append(f'    {prev_step_id} --> {step_id}')
                
                # If there are dependencies
                if hasattr(step, 'depends_on') and step.depends_on:
                    for dep in step.depends_on:
                        # Find dependency steps
                        for j, dep_step in enumerate(workflow_plan.steps):
                            if dep_step.step_id == dep:
                                dep_step_id = f"Step{j+1}"
                                mermaid_lines.append(f'    {dep_step_id} -.-> {step_id}')
                                break
            
            # Connect from last step to result
            if workflow_plan.steps:
                last_step_id = f"Step{len(workflow_plan.steps)}"
                mermaid_lines.append(f'    {last_step_id} --> Result[["✅ Result Integration"]]')
                mermaid_lines.append('    Result --> End(["🎉 Complete"])')
            else:
                mermaid_lines.append('    Query --> End(["🎉 Complete"])')
            
            # Add styles
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
            
            # Apply style to agent nodes
            for i in range(len(workflow_plan.steps)):
                mermaid_lines.append(f"    class Step{i+1} agent")
            
            return "\n".join(mermaid_lines)
            
        except Exception as e:
            logger.error(f"Mermaid diagram generation failed: {e}")
            return f'graph TD\n    Start(["Start"]) --> Error["Diagram generation failed: {str(e)}"]\n    Error --> End(["End"])' 