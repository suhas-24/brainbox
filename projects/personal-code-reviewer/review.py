#!/usr/bin/env python3
"""
Personal Code Review Assistant
Real-time demonstration of AI Forge capabilities
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add AI Forge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common/project-templates/llm-app-template/src"))

from agents.base_agent import BaseAgent, AgentResponse, AgentContext
from core.llm_manager import LLMManager
from memory.short_term import ConversationMemory


class CodeSecurityAgent(BaseAgent):
    """Agent specialized in security analysis."""
    
    def __init__(self):
        super().__init__(
            name="SecurityAnalyzer",
            system_prompt="""You are a security-focused code reviewer. Analyze code for:
            - SQL injection vulnerabilities
            - XSS vulnerabilities  
            - Authentication/authorization issues
            - Input validation problems
            - Secrets in code
            - Unsafe API calls
            
            Provide specific, actionable security recommendations."""
        )
    
    async def _execute_task(self, code: str, context: AgentContext, **kwargs) -> AgentResponse:
        prompt = f"""Analyze this code for security issues:

```
{code}
```

Project context: {context.working_memory.get('project_type', 'unknown')}
Focus on critical security vulnerabilities that could lead to data breaches or system compromise."""

        try:
            response = await self._generate_response(prompt, context)
            return AgentResponse(
                content=response,
                success=True,
                metadata={"analysis_type": "security", "agent": "SecurityAnalyzer"}
            )
        except Exception as e:
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )


class CodePerformanceAgent(BaseAgent):
    """Agent specialized in performance analysis."""
    
    def __init__(self):
        super().__init__(
            name="PerformanceAnalyzer", 
            system_prompt="""You are a performance-focused code reviewer. Analyze code for:
            - Inefficient algorithms
            - Memory leaks
            - Database query optimization
            - Caching opportunities
            - Resource usage
            - Scalability issues
            
            Provide specific performance optimization suggestions."""
        )
    
    async def _execute_task(self, code: str, context: AgentContext, **kwargs) -> AgentResponse:
        prompt = f"""Analyze this code for performance issues:

```
{code}
```

Project context: {context.working_memory.get('project_type', 'unknown')}
Language: {context.working_memory.get('language', 'unknown')}
Focus on optimizations that will have measurable impact."""

        try:
            response = await self._generate_response(prompt, context)
            return AgentResponse(
                content=response,
                success=True,
                metadata={"analysis_type": "performance", "agent": "PerformanceAnalyzer"}
            )
        except Exception as e:
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )


class CodeStyleAgent(BaseAgent):
    """Agent specialized in code style and best practices."""
    
    def __init__(self):
        super().__init__(
            name="StyleAnalyzer",
            system_prompt="""You are a code style and best practices reviewer. Analyze code for:
            - Code organization and structure
            - Naming conventions
            - Documentation and comments
            - Error handling
            - Code duplication
            - Language-specific best practices
            
            Provide suggestions for cleaner, more maintainable code."""
        )
    
    async def _execute_task(self, code: str, context: AgentContext, **kwargs) -> AgentResponse:
        prompt = f"""Analyze this code for style and best practices:

```
{code}
```

Project context: {context.working_memory.get('project_type', 'unknown')}
Language: {context.working_memory.get('language', 'unknown')}
User experience level: {context.working_memory.get('user_level', 'intermediate')}
Focus on maintainability and readability improvements."""

        try:
            response = await self._generate_response(prompt, context)
            return AgentResponse(
                content=response,
                success=True,
                metadata={"analysis_type": "style", "agent": "StyleAnalyzer"}
            )
        except Exception as e:
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )


class PersonalCodeReviewer:
    """Main code review system using AI Forge framework."""
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.llm_manager = LLMManager()
        
        # Initialize specialized agents
        self.security_agent = CodeSecurityAgent()
        self.performance_agent = CodePerformanceAgent()
        self.style_agent = CodeStyleAgent()
        
        # Load user preferences
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file."""
        prefs_file = Path("user_preferences.json")
        if prefs_file.exists():
            with open(prefs_file) as f:
                return json.load(f)
        
        # Default preferences
        return {
            "focus_areas": ["security", "performance", "style"],
            "verbosity": "detailed",
            "project_types": {},
            "feedback_history": []
        }
    
    def _save_preferences(self):
        """Save user preferences to file."""
        with open("user_preferences.json", "w") as f:
            json.dump(self.preferences, f, indent=2)
    
    def _detect_project_context(self, file_path: str, code: str) -> Dict[str, Any]:
        """Detect project type and context from file and code."""
        context = {
            "language": self._detect_language(file_path),
            "project_type": "unknown",
            "framework": None,
            "user_level": "intermediate"
        }
        
        # Detect framework/project type
        if "fastapi" in code.lower() or "from fastapi" in code:
            context["project_type"] = "fastapi_backend"
            context["framework"] = "FastAPI"
        elif "react" in code.lower() or "jsx" in file_path:
            context["project_type"] = "react_frontend"
            context["framework"] = "React"
        elif "pandas" in code or "numpy" in code:
            context["project_type"] = "data_analysis"
            context["framework"] = "Data Science"
        elif "def test_" in code or "import pytest" in code:
            context["project_type"] = "testing"
            context["framework"] = "Testing"
        
        return context
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c"
        }
        return language_map.get(ext, "unknown")
    
    async def review_code(self, file_path: str, code: str = None) -> Dict[str, Any]:
        """Main code review function."""
        
        # Read code if not provided
        if code is None:
            if not os.path.exists(file_path):
                return {"error": f"File {file_path} not found"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        
        # Detect project context
        project_context = self._detect_project_context(file_path, code)
        
        # Create session context
        session_id = f"review_{Path(file_path).stem}"
        context = AgentContext(
            session_id=session_id,
            working_memory=project_context
        )
        
        # Store context in memory
        for key, value in project_context.items():
            self.memory.set_working_memory(session_id, key, value)
        
        # Run analysis based on user preferences
        results = {}
        focus_areas = self.preferences.get("focus_areas", ["security", "performance", "style"])
        
        print(f"🔍 Analyzing {file_path} ({project_context['language']} - {project_context['project_type']})...")
        
        # Security analysis
        if "security" in focus_areas:
            print("  🔒 Running security analysis...")
            security_result = await self.security_agent.execute(code, context)
            results["security"] = security_result
        
        # Performance analysis
        if "performance" in focus_areas:
            print("  ⚡ Running performance analysis...")
            performance_result = await self.performance_agent.execute(code, context)
            results["performance"] = performance_result
        
        # Style analysis
        if "style" in focus_areas:
            print("  ✨ Running style analysis...")
            style_result = await self.style_agent.execute(code, context)
            results["style"] = style_result
        
        # Compile final report
        report = self._compile_report(results, project_context)
        
        # Remember this review for learning
        self._record_review(file_path, project_context, report)
        
        return report
    
    def _compile_report(self, results: Dict[str, AgentResponse], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile analysis results into a comprehensive report."""
        
        report = {
            "file_info": context,
            "summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "suggestions": 0
            },
            "analyses": {},
            "recommendations": []
        }
        
        for analysis_type, result in results.items():
            if result.success:
                report["analyses"][analysis_type] = {
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata
                }
                
                # Simple issue counting (in real implementation, would parse AI response)
                content_lower = result.content.lower()
                if "critical" in content_lower or "security" in content_lower:
                    report["summary"]["critical_issues"] += 1
                if "issue" in content_lower or "problem" in content_lower:
                    report["summary"]["total_issues"] += 1
                if "suggest" in content_lower or "recommend" in content_lower:
                    report["summary"]["suggestions"] += 1
            else:
                report["analyses"][analysis_type] = {
                    "error": result.error
                }
        
        return report
    
    def _record_review(self, file_path: str, context: Dict[str, Any], report: Dict[str, Any]):
        """Record this review for learning and improvement."""
        review_record = {
            "timestamp": "2024-01-01T00:00:00",  # In real implementation, use actual timestamp
            "file_path": file_path,
            "context": context,
            "summary": report["summary"]
        }
        
        self.preferences["feedback_history"].append(review_record)
        self._save_preferences()
    
    def print_report(self, report: Dict[str, Any]):
        """Print a formatted report to console."""
        print("\n" + "="*60)
        print("🤖 AI FORGE PERSONAL CODE REVIEW REPORT")
        print("="*60)
        
        # Summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"  • Total Issues: {summary['total_issues']}")
        print(f"  • Critical Issues: {summary['critical_issues']}")
        print(f"  • Suggestions: {summary['suggestions']}")
        
        # File info
        file_info = report["file_info"]
        print(f"\n📁 FILE INFO:")
        print(f"  • Language: {file_info['language']}")
        print(f"  • Project Type: {file_info['project_type']}")
        if file_info.get('framework'):
            print(f"  • Framework: {file_info['framework']}")
        
        # Detailed analyses
        for analysis_type, analysis in report["analyses"].items():
            if "error" in analysis:
                print(f"\n❌ {analysis_type.upper()} ANALYSIS FAILED:")
                print(f"  Error: {analysis['error']}")
            else:
                print(f"\n🔍 {analysis_type.upper()} ANALYSIS:")
                print(f"  {analysis['content']}")
                print(f"  (Completed in {analysis['execution_time']:.2f}s)")
        
        print("\n" + "="*60)


async def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Personal Code Review Assistant")
    parser.add_argument("--file", required=True, help="File to review")
    parser.add_argument("--focus", nargs="+", 
                       choices=["security", "performance", "style"],
                       default=["security", "performance", "style"],
                       help="Focus areas for review")
    
    args = parser.parse_args()
    
    # Initialize reviewer
    reviewer = PersonalCodeReviewer()
    reviewer.preferences["focus_areas"] = args.focus
    
    try:
        # Run review
        report = await reviewer.review_code(args.file)
        
        if "error" in report:
            print(f"❌ Error: {report['error']}")
            return 1
        
        # Print results
        reviewer.print_report(report)
        
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
