# prompt_templates.py
"""
Prompt template system for different datasets
Handles system prompts, user queries, and expected output formats
"""

from typing import Tuple, Dict, Any


PROMPT_TEMPLATES = {
    "mmlu": {
        "system": "You are a knowledgeable assistant. Answer the multiple choice question concisely.",
        "format": """{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer: """,
        "expected_format": "[A|B|C|D]",
        "instructions": "Respond with only the letter (A, B, C, or D) of the correct answer."
    },
    
    "gsm8k": {
        "system": "You are a math expert. Solve math problems step by step. Always show your work.",
        "format": """{question}

Let me solve this step by step:
""",
        "expected_format": "number",
        "instructions": "Solve the problem step by step. End with the final answer."
    },
    
    "sharegpt": {
        "system": "You are a helpful, harmless, and honest assistant.",
        "format": "{prompt}",
        "expected_format": "free text",
        "instructions": "Provide a helpful response."
    }
}


def build_prompt(example: Dict[str, Any], dataset_type: str) -> Tuple[str, str, str]:
    """
    Build formatted prompt with system message and user query
    
    Args:
        example: Example dict with 'prompt' and 'answer' keys
        dataset_type: 'mmlu', 'gsm8k', or 'sharegpt'
    
    Returns:
        (system_prompt, user_prompt, expected_answer)
    """
    if dataset_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    template = PROMPT_TEMPLATES[dataset_type]
    system_prompt = template["system"]
    
    # Parse prompt based on dataset type
    if dataset_type == "mmlu":
        # Split MMLU prompt into question and choices
        lines = example["prompt"].split('\n')
        question = lines[0]
        
        # Extract choices (they are in format "A) ...", "B) ...", etc)
        choices = [line[3:] if len(line) > 3 else "" for line in lines[1:5]]
        while len(choices) < 4:
            choices.append("")
        
        user_prompt = template["format"].format(
            question=question,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3]
        )
    
    elif dataset_type == "gsm8k":
        user_prompt = template["format"].format(question=example["prompt"])
    
    else:  # sharegpt
        user_prompt = template["format"].format(prompt=example["prompt"])
    
    expected_answer = example.get("answer", "")
    
    return system_prompt, user_prompt, expected_answer


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
    
    system, user, answer = build_prompt(test_mmlu, "mmlu")
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
    
    system, user, answer = build_prompt(test_gsm8k, "gsm8k")
    print("GSM8K Example:")
    print(f"System: {system}")
    print(f"User:\n{user}")
    print(f"Answer: {answer}")
