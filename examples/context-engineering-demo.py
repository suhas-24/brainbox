"""
Context Engineering in Action: Real-World Example

This demonstrates how context engineering transforms basic AI interactions
into sophisticated, context-aware applications.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any


class ContextEngineeringDemo:
    """Demonstrates the power of context engineering with real examples."""
    
    def __init__(self):
        self.user_profiles = {
            "john_doe": {
                "role": "senior_developer",
                "team": "backend_engineering",
                "experience_level": "expert",
                "preferred_languages": ["python", "go"],
                "current_projects": ["payment_service", "user_auth"],
                "working_hours": "9am-6pm PST",
                "communication_style": "technical_detailed"
            },
            "sarah_chen": {
                "role": "product_manager",
                "team": "product_strategy",
                "experience_level": "senior",
                "preferred_languages": ["none"],
                "current_projects": ["mobile_app_redesign", "user_onboarding"],
                "working_hours": "8am-5pm EST",
                "communication_style": "business_focused"
            }
        }
        
        self.project_context = {
            "payment_service": {
                "status": "in_development",
                "tech_stack": ["python", "fastapi", "postgresql", "redis"],
                "team_size": 5,
                "deadline": "2024-03-15",
                "critical_features": ["fraud_detection", "multi_currency", "webhooks"],
                "recent_issues": ["high_latency", "memory_leaks"]
            },
            "mobile_app_redesign": {
                "status": "design_phase",
                "platform": ["ios", "android", "web"],
                "team_size": 8,
                "deadline": "2024-04-30",
                "user_research": ["usability_study", "a_b_tests"],
                "key_metrics": ["user_retention", "conversion_rate"]
            }
        }
    
    async def basic_response_without_context(self, query: str) -> str:
        """Example of basic AI response without context engineering."""
        # This is what most basic AI implementations look like
        return f"I can help you with {query}. What specific information do you need?"
    
    async def advanced_response_with_context(
        self, 
        query: str, 
        user_id: str,
        session_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Example of context-engineered response."""
        
        # 1. CONTEXT RETRIEVAL
        user_profile = self.user_profiles.get(user_id, {})
        current_time = datetime.now()
        
        # 2. CONTEXT ASSEMBLY
        context = {
            "user": user_profile,
            "temporal": {
                "current_time": current_time.isoformat(),
                "day_of_week": current_time.strftime("%A"),
                "is_business_hours": self._is_business_hours(user_profile, current_time)
            },
            "session": session_context or {},
            "projects": self._get_relevant_projects(user_profile),
            "conversation_history": self._get_conversation_history(user_id)
        }
        
        # 3. CONTEXT ADAPTATION
        adapted_response = self._adapt_response_to_context(query, context)
        
        # 4. CONTEXT COMPRESSION (for token efficiency)
        compressed_context = self._compress_context(context, query)
        
        return {
            "response": adapted_response,
            "context_used": compressed_context,
            "reasoning": self._explain_context_decisions(context, query),
            "suggested_actions": self._generate_suggested_actions(context, query)
        }
    
    def _is_business_hours(self, user_profile: Dict, current_time: datetime) -> bool:
        """Check if it's within user's business hours."""
        # Simplified implementation
        hour = current_time.hour
        return 9 <= hour <= 17
    
    def _get_relevant_projects(self, user_profile: Dict) -> List[Dict]:
        """Get projects relevant to the user."""
        user_projects = user_profile.get("current_projects", [])
        return [
            {**self.project_context[proj], "name": proj} 
            for proj in user_projects 
            if proj in self.project_context
        ]
    
    def _get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get recent conversation history."""
        # In real implementation, this would come from memory system
        return [
            {
                "timestamp": "2024-01-01T10:00:00",
                "user": "Can you help me debug the payment service?",
                "assistant": "I can help you debug the payment service. What specific issue are you experiencing?"
            }
        ]
    
    def _adapt_response_to_context(self, query: str, context: Dict) -> str:
        """Adapt response based on comprehensive context."""
        user = context["user"]
        projects = context["projects"]
        
        # Role-based adaptation
        if user.get("role") == "senior_developer":
            if "debug" in query.lower() and projects:
                project = projects[0]  # Most recent project
                return f"""Based on your current work on {project['name']}, I can help you debug the issue. 
                
Given the recent {', '.join(project.get('recent_issues', []))} issues in this project, here's what I'd check first:

1. **Performance Analysis**: Check the {', '.join(project['tech_stack'])} stack for bottlenecks
2. **Error Logs**: Review recent logs for patterns related to {', '.join(project['critical_features'])}
3. **Resource Usage**: Monitor memory and CPU usage given the memory leak concerns

What specific symptoms are you observing?"""
        
        elif user.get("role") == "product_manager":
            if "status" in query.lower() and projects:
                project = projects[0]
                return f"""Here's the current status of {project['name']}:

**Phase**: {project['status'].replace('_', ' ').title()}
**Timeline**: On track for {project['deadline']} deadline
**Team**: {project['team_size']} team members
**Key Metrics**: Tracking {', '.join(project.get('key_metrics', []))}

Would you like a detailed breakdown of any specific area?"""
        
        return f"I can help you with {query}. What specific information do you need?"
    
    def _compress_context(self, context: Dict, query: str) -> Dict:
        """Compress context to essential information only."""
        # This is crucial for token efficiency
        return {
            "user_role": context["user"].get("role"),
            "relevant_projects": [p["name"] for p in context["projects"][:2]],
            "is_urgent": not context["temporal"]["is_business_hours"],
            "communication_style": context["user"].get("communication_style")
        }
    
    def _explain_context_decisions(self, context: Dict, query: str) -> List[str]:
        """Explain why certain context was used."""
        decisions = []
        
        user = context["user"]
        if user.get("role") == "senior_developer":
            decisions.append("Used technical communication style based on user role")
        
        if context["projects"]:
            decisions.append(f"Referenced current project: {context['projects'][0]['name']}")
        
        if not context["temporal"]["is_business_hours"]:
            decisions.append("Noted after-hours timing for potential urgency")
        
        return decisions
    
    def _generate_suggested_actions(self, context: Dict, query: str) -> List[str]:
        """Generate contextually relevant suggested actions."""
        suggestions = []
        
        user = context["user"]
        projects = context["projects"]
        
        if "debug" in query.lower() and projects:
            project = projects[0]
            suggestions.extend([
                f"Check {project['name']} error logs",
                f"Review {project['tech_stack'][0]} performance metrics",
                "Run diagnostic tests on critical features"
            ])
        
        elif user.get("role") == "product_manager":
            suggestions.extend([
                "Schedule team standup",
                "Review user feedback",
                "Update stakeholder dashboard"
            ])
        
        return suggestions


async def demonstrate_context_engineering():
    """Run the demonstration."""
    demo = ContextEngineeringDemo()
    
    query = "Can you help me debug the payment service?"
    
    print("🔍 CONTEXT ENGINEERING DEMONSTRATION")
    print("=" * 50)
    
    # Without context
    print("\n❌ WITHOUT CONTEXT ENGINEERING:")
    basic_response = await demo.basic_response_without_context(query)
    print(f"Response: {basic_response}")
    print("Issues: Generic, unhelpful, requires multiple follow-ups")
    
    # With context engineering
    print("\n✅ WITH CONTEXT ENGINEERING:")
    advanced_response = await demo.advanced_response_with_context(
        query, 
        "john_doe",
        {"urgency": "high", "previous_attempts": ["restarted_service", "checked_logs"]}
    )
    
    print(f"Response: {advanced_response['response']}")
    print(f"\nContext Used: {advanced_response['context_used']}")
    print(f"\nReasoning: {advanced_response['reasoning']}")
    print(f"\nSuggested Actions: {advanced_response['suggested_actions']}")
    
    print("\n📊 BENEFITS:")
    print("✓ Personalized to user's role and expertise")
    print("✓ Considers current project context")
    print("✓ Suggests specific, actionable next steps")
    print("✓ Adapts communication style")
    print("✓ Reduces back-and-forth conversations")


async def real_world_scenarios():
    """Show more real-world scenarios."""
    demo = ContextEngineeringDemo()
    
    scenarios = [
        {
            "user": "sarah_chen",
            "query": "What's the status of our mobile app project?",
            "context": {"meeting_prep": True, "stakeholders": ["ceo", "design_team"]}
        },
        {
            "user": "john_doe", 
            "query": "The payment service is throwing 500 errors",
            "context": {"urgency": "critical", "production_impact": True}
        }
    ]
    
    print("\n🌟 REAL-WORLD SCENARIOS")
    print("=" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"User: {scenario['user']}")
        print(f"Query: {scenario['query']}")
        
        response = await demo.advanced_response_with_context(
            scenario['query'],
            scenario['user'],
            scenario['context']
        )
        
        print(f"Context-Aware Response: {response['response'][:200]}...")
        print(f"Key Context Used: {response['context_used']}")


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(demonstrate_context_engineering())
    asyncio.run(real_world_scenarios())
