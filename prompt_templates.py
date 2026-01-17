# prompt_templates.py
"""
Prompt template system for different datasets
Handles system prompts, user queries, and expected output formats
"""

from typing import Tuple, Dict, Any


PROMPT_TEMPLATES = {
    "mmlu": {
        "system": """You are an expert assistant that solves multiple-choice questions using step-by-step reasoning.
IMPORTANT: Always think step-by-step before answering. Format your final answer as exactly "ANSWER: [A/B/C/D]". Do NOT add extra text after the answer.""",
        "user_template": """Question: {question}

Options:
A) {choice_a}
B) {choice_b} 
C) {choice_c}
D) {choice_d}

Instructions:
1. Read the question carefully
2. Analyze each option
3. Think step-by-step about which is correct
4. Give your final answer as exactly "ANSWER: [A/B/C/D]"

Reasoning:""",
        "expected_format": "ANSWER: [A|B|C|D]",
        "parser": "extract_mmlu_letter",
        "max_tokens": 128
    },
    
    "gsm8k": {
        "system": """You are a math expert who solves word problems step-by-step. 
Show ALL your reasoning and calculations clearly. 
At the end, box your final numerical answer as "FINAL_ANSWER: [number]".

Format requirements:
- Explain each step
- Show all calculations  
- End with exactly "FINAL_ANSWER: [number]"
- No other text after the box.""",
        "user_template": """Problem: {question}

Solve this step-by-step and show your work.

Step 1:""",
        "expected_format": "FINAL_ANSWER: [number]",
        "parser": "extract_gsm8k_number",
        "max_tokens": 256
    }
}



def build_improved_prompt(example: Dict[str, Any], dataset_type: str) -> Tuple[str, str, str]:
    """Build CoT-enabled prompt with structured output"""
    template = PROMPT_TEMPLATES[dataset_type]
    system_prompt = template["system"]
    
    if dataset_type == "mmlu":
        # Parse MMLU format more robustly
        lines = example["prompt"].split('\n')
        question = lines[0].strip()
        
        # Extract choices more reliably
        choices = {}
        for i, line in enumerate(lines[1:], 1):
            if line.strip() and len(line) > 2:
                choices[chr(ord('A') + i-1)] = line[3:].strip() if len(line) > 3 else f"Option {i}"
        
        user_prompt = template["user_template"].format(
            question=question,
            choice_a=choices.get('A', ''),
            choice_b=choices.get('B', ''),
            choice_c=choices.get('C', ''),
            choice_d=choices.get('D', '')
        )
    
    elif dataset_type == "gsm8k":
        user_prompt = template["user_template"].format(
            question=example["prompt"]
        )
    
    return system_prompt, user_prompt, example.get("answer", "")


def get_prompt_instructions(dataset_type: str) -> str:
    """Get instructions for a dataset type"""
    if dataset_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return PROMPT_TEMPLATES[dataset_type]["instructions"]


def get_expected_format(dataset_type: str) -> str:
    """Get expected output format for a dataset type"""
    if dataset_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return PROMPT_TEMPLATES[dataset_type]["expected_format"]


if __name__ == "__main__":
    # Test prompt building
    test_mmlu = {
        "dataset": "mmlu",
        "prompt": "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Rome",
        "answer": "B",
        "difficulty": "easy"
    }
    
    system, user, answer = build_improved_prompt(test_mmlu, "mmlu")
    print("MMLU Example:")
    print(f"System: {system}")
    print(f"User:\n{user}")
    print(f"Answer: {answer}\n")
    
    test_gsm8k = {
        "dataset": "gsm8k",
        "prompt": "If John has 5 apples and gives away 2, how many does he have?",
        "answer": "3",
        "difficulty": "easy"
    }
    
    system, user, answer = build_improved_prompt(test_gsm8k, "gsm8k")
    print("GSM8K Example:")
    print(f"System: {system}")
    print(f"User:\n{user}")
    print(f"Answer: {answer}")
