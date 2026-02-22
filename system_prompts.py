"""
System Prompt Manager
====================
Manage system prompts for different use cases
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SystemPrompt:
    """System prompt template"""
    name: str
    description: str
    template: str
    variables: List[str]


# Pre-built system prompts
SYSTEM_PROMPTS = {
    # Coding prompts
    "code_reviewer": SystemPrompt(
        name="Code Reviewer",
        description="Review code for bugs and improvements",
        template="""You are an expert code reviewer. Review the following code for:
1. Bugs and potential issues
2. Security vulnerabilities
3. Performance problems
4. Code style improvements
5. Best practices violations

Provide a detailed report with specific suggestions.

Code to review:
{code}

Review:""",
        variables=["code"]
    ),
    
    "code_explainer": SystemPrompt(
        name="Code Explainer",
        description="Explain code in simple terms",
        template="""You are a programming teacher. Explain the following code in simple terms that a beginner can understand. Use analogies where helpful.

Code:
{code}

Explanation:""",
        variables=["code"]
    ),
    
    "code_writer": SystemPrompt(
        name="Code Writer",
        description="Write code based on requirements",
        template="""Write {language} code that fulfills these requirements:

{requirements}

Requirements:
{requirements}

Additional context:
{context}

Code:""",
        variables=["language", "requirements", "context"]
    ),
    
    # Writing prompts
    "summarizer": SystemPrompt(
        name="Summarizer",
        description="Summarize long text",
        template="""Summarize the following text in {length} sentences or less. Include the main points and key details.

Text:
{text}

Summary:""",
        variables=["length", "text"]
    ),
    
    "blog_writer": SystemPrompt(
        name="Blog Writer",
        description="Write engaging blog posts",
        template="""Write an engaging blog post about {topic}. 

Requirements:
- Tone: {tone}
- Length: {length}
- Include: {include_points}

Topic: {topic}

Blog Post:""",
        variables=["topic", "tone", "length", "include_points"]
    ),
    
    # Analysis prompts
    "data_analyzer": SystemPrompt(
        name="Data Analyzer",
        description="Analyze data and provide insights",
        template="""Analyze the following data and provide insights:

Data:
{data}

Analysis Type: {analysis_type}
Focus Areas: {focus_areas}

Analysis:""",
        variables=["data", "analysis_type", "focus_areas"]
    ),
    
    # Customer service
    "support_agent": SystemPrompt(
        name="Support Agent",
        description="Customer support responses",
        template="""You are a helpful customer support agent. Respond to the following customer inquiry professionally and empathetically.

Customer Query:
{query}

Previous Context:
{context}

Response:""",
        variables=["query", "context"]
    ),
    
    # Education
    "tutor": SystemPrompt(
        name="Tutor",
        description="Educational tutoring",
        template="""You are a patient tutor teaching {subject}. 

Student's question:
{question}

Student's level: {level}

Explain in a way that helps the student understand. Ask clarifying questions if needed.

Explanation:""",
        variables=["subject", "question", "level"]
    ),
}


class PromptManager:
    """
    Manage and use system prompts
    
    Usage:
        manager = PromptManager()
        
        # Get a prompt
        prompt = manager.get_prompt("code_reviewer")
        
        # Fill in variables
        filled = prompt.fill(code="def foo(): pass")
        
        # Generate
        response = await llm.generate(filled)
    """
    
    def __init__(self):
        self.prompts = SYSTEM_PROMPTS.copy()
    
    def get_prompt(self, name: str) -> Optional[SystemPrompt]:
        """Get a system prompt by name"""
        return self.prompts.get(name)
    
    def list_prompts(self) -> List[str]:
        """List all available prompts"""
        return list(self.prompts.keys())
    
    def add_prompt(self, prompt: SystemPrompt):
        """Add a custom prompt"""
        self.prompts[prompt.name.lower().replace(" ", "_")] = prompt
    
    def create_prompt(
        self,
        name: str,
        description: str,
        template: str,
        variables: List[str]
    ):
        """Create and add a new prompt"""
        prompt = SystemPrompt(
            name=name,
            description=description,
            template=template,
            variables=variables
        )
        self.add_prompt(prompt)
    
    def render_prompt(self, name: str, **kwargs) -> str:
        """Render a prompt with variables filled"""
        prompt = self.get_prompt(name)
        
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found")
        
        return prompt.template.format(**kwargs)
    
    def get_prompts_by_category(self, category: str) -> List[SystemPrompt]:
        """Get prompts by category"""
        # Simple categorization
        coding = ["code_reviewer", "code_explainer", "code_writer"]
        writing = ["summarizer", "blog_writer"]
        analysis = ["data_analyzer"]
        support = ["support_agent"]
        education = ["tutor"]
        
        categories = {
            "coding": coding,
            "writing": writing,
            "analysis": analysis,
            "support": support,
            "education": education
        }
        
        names = categories.get(category.lower(), [])
        return [self.prompts[n] for n in names if n in self.prompts]


# Usage example
def example_prompt_manager():
    """Example usage of PromptManager"""
    
    manager = PromptManager()
    
    # List all prompts
    print("Available prompts:")
    for name in manager.list_prompts():
        prompt = manager.get_prompt(name)
        print(f"  - {name}: {prompt.description}")
    
    # Use a prompt
    print("\nUsing 'code_reviewer':")
    prompt = manager.render_prompt(
        "code_reviewer",
        code="function add(a,b){return a+b;}"
    )
    print(prompt[:200] + "...")


if __name__ == "__main__":
    example_prompt_manager()
