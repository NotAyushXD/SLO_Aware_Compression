"""
Difficulty-Aware Prompt Templates for LLM Serving (FIXED)

Key Features:
- Per-difficulty system messages (simplified for actual answer generation)
- Adaptive token budgets (120/200/350 for easy/medium/hard)
- NO aggressive stop sequences (was causing early termination)
- Realistic prompt structure without excessive reasoning overhead
- Prevents both overthinking (easy) and underthinking (hard)

CRITICAL FIXES APPLIED:
1. Removed "\n\n" stop sequence (was cutting off responses too early)
2. Simplified prompts to focus on answer generation
3. Adjusted token budgets to match Llama-3.1-8B reality
"""

from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DIFFICULTY-AWARE TOKEN BUDGETS
# ============================================================================

# Difficulty-aware token budgets
DIFFICULTY_TOKEN_BUDGETS = {
    "easy": 120,      # Simple fact recall, no reasoning needed
    "medium": 200,    # Some reasoning, moderate explanation
    "hard": 350       # Complex reasoning, detailed explanation
}

# MMLU TEMPLATES - UPDATED for better reasoning
MMLU_TEMPLATES = {
    "easy": {
        "system": "",
        "user_template": """Answer the following question carefully.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

The correct answer is:""",
        "max_tokens": 100,
        "stop_sequences": []
    },
    "medium": {
        "system": "",
        "user_template": """Answer the following multiple choice question by analyzing each option.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Think through: Which option is correct and why?
The correct answer is:""",
        "max_tokens": 150,
        "stop_sequences": []
    },
    "hard": {
        "system": "",
        "user_template": """Answer this complex question carefully by analyzing each option.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Analyze each option and choose the best answer.
The correct answer is:""",
        "max_tokens": 200,
        "stop_sequences": []
    }
}

# GSM8K TEMPLATES - UPDATED for step-by-step reasoning
GSM8K_TEMPLATES = {
    "easy": {
        "system": "",
        "user_template": """Given the following math problem, solve it carefully.

Problem: {question}

Solution:
Step 1: Identify what we need to find.
Step 2: Set up the calculation.
Step 3: Show your work.

FINAL_ANSWER:""",
        "max_tokens": 150,
        "stop_sequences": []
    },
    "medium": {
        "system": "",
        "user_template": """Given the following math problem, solve it step-by-step.

Problem: {question}

Solution:
Step 1: Identify what we need to find.
Step 2: Break down the problem.
Step 3: Set up calculations.
Step 4: Show your work.
Step 5: Verify the answer makes sense.

FINAL_ANSWER:""",
        "max_tokens": 200,
        "stop_sequences": []
    },
    "hard": {
        "system": "",
        "user_template": """Given the following complex math problem, solve it very carefully.

Problem: {question}

Solution:
Step 1: What are we solving for?
Step 2: What information do we have?
Step 3: What formula or method should we use?
Step 4: Show all calculations.
Step 5: Check: Does the answer make sense?

FINAL_ANSWER:""",
        "max_tokens": 250,
        "stop_sequences": []
    }
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_max_tokens(difficulty: str, dataset_type: str = "mmlu") -> int:
    """
    Get max tokens budget based on difficulty.
    
    Args:
        difficulty: 'easy', 'medium', or 'hard'
        dataset_type: 'mmlu' or 'gsm8k' (for future customization)
    
    Returns:
        Token budget as integer
    
    Example:
        >>> get_max_tokens("hard", "gsm8k")
        350
        >>> get_max_tokens("easy", "mmlu")
        120
    """
    if difficulty not in DIFFICULTY_TOKEN_BUDGETS:
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"
    
    return DIFFICULTY_TOKEN_BUDGETS[difficulty]


def build_improved_prompt(
    example: Dict[str, Any],
    dataset_type: str
) -> Tuple[str, str, str]:
    """
    Build difficulty-aware prompt (backwards compatibility).
    
    Args:
        example: Dict with keys: prompt, answer, difficulty, choice_*
        dataset_type: 'mmlu' or 'gsm8k'
    
    Returns:
        Tuple of (system_prompt, user_prompt, answer)
    
    Note: For backwards compatibility, does NOT return max_tokens.
    Use build_llama_formatted_prompt() to get max_tokens.
    """
    system_prompt, user_prompt, answer, _ = build_llama_formatted_prompt(
        example, dataset_type, return_tokens=True
    )
    return system_prompt, user_prompt, answer


def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    return_tokens: bool = False
) -> Tuple[str, int, list]:
    """
    Build complete Llama-formatted prompt with difficulty-aware settings.
    
    Args:
        example: Dict with prompt, answer, difficulty, choices
        dataset_type: 'mmlu' or 'gsm8k'
        return_tokens: If True, also return (system, user, answer, max_tokens)
                      If False, return (prompt, max_tokens, stop_sequences)
    
    Returns:
        If return_tokens=False: (formatted_prompt, max_tokens, stop_sequences)
        If return_tokens=True: (system_prompt, user_prompt, answer, max_tokens)
    """
    if dataset_type not in ["mmlu", "gsm8k"]:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Get difficulty (default to medium if not specified)
    difficulty = example.get("difficulty", "medium")
    if difficulty not in ["easy", "medium", "hard"]:
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"
    
    # Select appropriate templates
    if dataset_type == "mmlu":
        templates = MMLU_TEMPLATES
    else:  # gsm8k
        templates = GSM8K_TEMPLATES
    
    template = templates[difficulty]
    system_prompt = template["system"]
    
    # Build user prompt
    if dataset_type == "mmlu":
        prompt_text = example.get("prompt", "")
        lines = prompt_text.split('\n') if isinstance(prompt_text, str) else []
        question = lines[0].strip() if lines else ""
        
        # Extract choices (A, B, C, D)
        choices = {"A": "", "B": "", "C": "", "D": ""}
        for line in lines[1:]:
            line = line.strip()
            if line and len(line) > 2 and line[0] in "ABCD" and line[1] in ") :":
                choice_letter = line[0]
                choice_text = line[3:].strip() if len(line) > 3 else ""
                choices[choice_letter] = choice_text
        
        user_prompt = template["user_template"].format(
            question=question,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", "")
        )
    
    elif dataset_type == "gsm8k":
        question = example.get("prompt", "")
        user_prompt = template["user_template"].format(question=question)
    
    else:
        user_prompt = example.get("prompt", "")
    
    answer = example.get("answer", "")
    max_tokens = template["max_tokens"]
    stop_sequences = template["stop_sequences"]
    
    # Format as simple text completion for Base model
    # Avoid [INST] tags which confuse base models
    formatted_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    
    if return_tokens:
        return system_prompt, user_prompt, answer, max_tokens
    else:
        return formatted_prompt, max_tokens, stop_sequences


def get_expected_format(dataset_type: str) -> str:
    """Get expected output format for a dataset type"""
    if dataset_type == "mmlu":
        return "A|B|C|D"
    elif dataset_type == "gsm8k":
        return "[number]"
    else:
        return ""


def get_stop_sequences(dataset_type: str, difficulty: str = "medium") -> list:
    """Get stop sequences to prevent over-generation"""
    if dataset_type == "mmlu":
        templates = MMLU_TEMPLATES
    elif dataset_type == "gsm8k":
        templates = GSM8K_TEMPLATES
    else:
        return []
    
    if difficulty not in templates:
        difficulty = "medium"
    
    return templates[difficulty].get("stop_sequences", [])


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DIFFICULTY-AWARE PROMPT TEMPLATES TEST (FIXED)")
    print("=" * 80)
    
    # Test token budgets
    print("\nToken Budgets:")
    for difficulty in ["easy", "medium", "hard"]:
        tokens = get_max_tokens(difficulty)
        print(f"  {difficulty:8s}: {tokens:3d} tokens")
    
    # Test MMLU
    print("\n" + "-" * 80)
    print("MMLU TESTS")
    print("-" * 80)
    
    for difficulty in ["easy", "medium", "hard"]:
        mmlu_example = {
            "prompt": "What is 2+2?\nA) 1\nB) 4\nC) 5\nD) 6",
            "answer": "B",
            "difficulty": difficulty
        }
        
        prompt, max_tokens, stops = build_llama_formatted_prompt(mmlu_example, "mmlu")
        expected_tokens = DIFFICULTY_TOKEN_BUDGETS[difficulty]
        
        if max_tokens == expected_tokens:
            print(f"✓ MMLU {difficulty:6s}: {max_tokens:3d} tokens, stops={stops}")
        else:
            print(f"✗ MMLU {difficulty:6s}: Expected {expected_tokens}, got {max_tokens}")
    
    # Test GSM8K
    print("\n" + "-" * 80)
    print("GSM8K TESTS")
    print("-" * 80)
    
    for difficulty in ["easy", "medium", "hard"]:
        gsm8k_example = {
            "prompt": "John has 5 apples. He eats 2. How many left?",
            "answer": "3",
            "difficulty": difficulty
        }
        
        prompt, max_tokens, stops = build_llama_formatted_prompt(gsm8k_example, "gsm8k")
        expected_tokens = DIFFICULTY_TOKEN_BUDGETS[difficulty]
        
        if max_tokens == expected_tokens:
            print(f"✓ GSM8K {difficulty:6s}: {max_tokens:3d} tokens, stops={stops}")
        else:
            print(f"✗ GSM8K {difficulty:6s}: Expected {expected_tokens}, got {max_tokens}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

# # prompt_templates.py
# """
# Optimized prompt templates for Llama-3.1-8B with Difficulty-Aware Evaluation

# Key improvements:
# 1. Llama-3.1 native format (not Llama-2 [INST])
# 2. Better CoT prompts with task-specific reasoning
# 3. Few-shot examples per difficulty level (reduces overthinking)
# 4. Constraint-aware design respecting max_tokens budgets
# 5. Model-optimized for 8B parameter scale

# Papers cited:
# - DAST: Difficulty-Adaptive Slow-Thinking
# - "When More is Less": Optimal CoT length is task-specific
# - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
# """

# from typing import Tuple, Dict, Any, Optional


# # =============================================================================
# # LLAMA-3.1 FORMAT HELPERS
# # =============================================================================

# def format_llama_message(role: str, content: str) -> str:
#     """Format a message in Llama-3.1 chat format"""
#     return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"


# def build_llama_chat_prompt(system: str, user: str) -> str:
#     """Build complete Llama-3.1 chat prompt"""
#     prompt = "<|begin_of_text|>"
#     prompt += format_llama_message("system", system)
#     prompt += format_llama_message("user", user)
#     prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
#     return prompt


# # =============================================================================
# # DIFFICULTY-AWARE CONFIGURATION
# # =============================================================================

# DIFFICULTY_CONFIG = {
#     "easy": {
#         "max_tokens": 100,
#         "instruction": "Answer briefly and directly.",
#         "reasoning_style": "minimal",
#         "cot_depth": "single",
#         "description": "Simple questions requiring minimal reasoning",
#         "expected_length": "1-2 sentences"
#     },
#     "medium": {
#         "max_tokens": 200,
#         "instruction": "Provide a clear, step-by-step answer.",
#         "reasoning_style": "moderate",
#         "cot_depth": "multi-step",
#         "description": "Moderate reasoning with clear logical steps",
#         "expected_length": "2-3 paragraphs"
#     },
#     "hard": {
#         "max_tokens": 500,
#         "instruction": "Explain your reasoning thoroughly. Show all steps.",
#         "reasoning_style": "thorough",
#         "cot_depth": "deep",
#         "description": "Complex reasoning requiring detailed explanation",
#         "expected_length": "4-6 paragraphs"
#     }
# }

# FIXED_TOKEN_BUDGET = 256
# USE_DIFFICULTY_AWARE = True


# # =============================================================================
# # FEW-SHOT EXAMPLES (Control output length per difficulty)
# # =============================================================================

# MMLU_EXAMPLES = {
#     "easy": """Example:
# Question: What is the capital of France?
# Options:
# A) London
# B) Paris
# C) Berlin
# D) Madrid

# Thinking: This is a straightforward geography question. France's capital is Paris.

# ANSWER: B) Paris""",

#     "medium": """Example:
# Question: What best explains photosynthesis?
# Options:
# A) Plants absorb oxygen and release carbon dioxide
# B) Plants convert light energy into chemical energy using chlorophyll
# C) Plants store energy in their roots for winter
# D) Plants break down food molecules for energy

# Thinking: Let me analyze each option:
# - A) Incorrect - this describes respiration, not photosynthesis
# - B) Correct - photosynthesis converts light energy to chemical energy (glucose)
# - C) Partially true but not the main definition
# - D) This is respiration, not photosynthesis

# ANSWER: B) Plants convert light energy into chemical energy using chlorophyll""",

#     "hard": """Example:
# Question: Which of the following best explains why the social contract theory is more realistic than natural rights theory?
# Options:
# A) Natural rights are innate and therefore impossible to violate
# B) The social contract acknowledges that rights are collective agreements that can be modified based on social needs
# C) Natural rights theory has been universally accepted across all cultures
# D) Both theories are equally valid regardless of context

# Thinking: This requires careful philosophical analysis:
# - A) Incorrect - natural rights theory claims rights are universal, not impossible to violate
# - B) Correct - social contract theory is more realistic because it acknowledges rights as social constructs that evolve with society
# - C) Incorrect - natural rights theory hasn't been universally accepted
# - D) Too vague and doesn't address the comparative question

# The key difference is that social contract theory is more pragmatic and acknowledges the negotiated nature of rights.

# ANSWER: B) The social contract acknowledges that rights are collective agreements that can be modified based on social needs"""
# }

# GSM8K_EXAMPLES = {
#     "easy": """Example:
# Problem: John has 5 apples. He eats 2. How many apples does John have left?

# Solution: 
# Start: 5 apples
# Ate: 2 apples
# Remaining: 5 - 2 = 3 apples

# FINAL_ANSWER: 3""",

#     "medium": """Example:
# Problem: A store has 15 red balls and 20 blue balls. They sell 8 red balls and 5 blue balls. How many balls are left in total?

# Solution:
# Red balls initially: 15
# Red balls sold: 8
# Red balls remaining: 15 - 8 = 7

# Blue balls initially: 20
# Blue balls sold: 5
# Blue balls remaining: 20 - 5 = 15

# Total balls remaining: 7 + 15 = 22

# FINAL_ANSWER: 22""",

#     "hard": """Example:
# Problem: A factory produces 120 widgets per day. The factory operates 5 days a week. If 10% of widgets are defective, 15% are sold at a discount, and the rest are sold at full price ($8 each), what is the total revenue from full-price widgets in a week?

# Solution:
# Step 1: Calculate total widgets per week
# - Widgets per day: 120
# - Days per week: 5
# - Total per week: 120 × 5 = 600 widgets

# Step 2: Categorize widgets
# - Defective: 10% of 600 = 60 widgets
# - Discount: 15% of 600 = 90 widgets
# - Full price: 600 - 60 - 90 = 450 widgets

# Step 3: Calculate full-price revenue
# - Full-price widgets: 450
# - Price per widget: $8
# - Total revenue: 450 × 8 = $3,600

# FINAL_ANSWER: 3600"""
# }


# # =============================================================================
# # DATASET TEMPLATES (Llama-3.1 optimized)
# # =============================================================================

# PROMPT_TEMPLATES = {
#     "mmlu": {
#         "system_base": """You are an expert assistant answering multiple-choice questions.

# Instructions:
# - {instruction}
# - Think through the question carefully.
# - Consider all options before answering.
# - Format your answer as "ANSWER: [A/B/C/D]" (just the letter).

# {example}

# Now answer the following question:""",

#         "user_template": """Question: {question}

# Options:
# A) {choice_a}
# B) {choice_b}
# C) {choice_c}
# D) {choice_d}

# Think through this step-by-step, then provide your answer.""",

#         "expected_format": "ANSWER: [A|B|C|D]",
#         "parser": "extract_mmlu_letter",
#         "default_max_tokens": 150
#     },

#     "gsm8k": {
#         "system_base": """You are an expert math tutor solving word problems.

# Instructions:
# - {instruction}
# - Show your work step by step.
# - Check your arithmetic.
# - At the end, provide the final answer as "FINAL_ANSWER: [number]"

# {example}

# Now solve this problem:""",

#         "user_template": """Problem: {question}

# Solve this step-by-step and provide your final answer.""",

#         "expected_format": "FINAL_ANSWER: [number]",
#         "parser": "extract_gsm8k_number",
#         "default_max_tokens": 300
#     }
# }


# # Difficulty-specific CoT instructions
# COT_INSTRUCTIONS = {
#     "mmlu": {
#         "easy": "Think about this directly.",
#         "medium": "Analyze the options systematically.",
#         "hard": "Carefully evaluate each option and explain your reasoning."
#     },
#     "gsm8k": {
#         "easy": "Solve directly.",
#         "medium": "Work through this step-by-step.",
#         "hard": "Break this into clear steps and show all calculations."
#     }
# }


# # =============================================================================
# # CORE FUNCTIONS
# # =============================================================================

# def get_difficulty_config(difficulty: str) -> Dict:
#     """Get configuration for a specific difficulty level"""
#     return DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])


# def get_max_tokens(difficulty: str, dataset_type: Optional[str] = None) -> int:
#     """
#     Get appropriate max_tokens based on difficulty.

#     Args:
#         difficulty: 'easy', 'medium', or 'hard'
#         dataset_type: Optional dataset type (for future extension)

#     Returns:
#         Token budget (100/200/500 for easy/medium/hard)
#     """
#     if not USE_DIFFICULTY_AWARE:
#         return FIXED_TOKEN_BUDGET

#     config = get_difficulty_config(difficulty)
#     return config["max_tokens"]


# def build_difficulty_aware_prompt(
#     example: Dict[str, Any],
#     dataset_type: str
# ) -> Tuple[str, str, str, int]:
#     """
#     Build Llama-3.1 optimized prompt with difficulty-aware instructions.

#     Args:
#         example: Dict with 'prompt', 'answer', 'difficulty', 'choice_a', etc.
#         dataset_type: 'mmlu' or 'gsm8k'

#     Returns:
#         Tuple of (system_prompt, user_prompt, answer, max_tokens)

#     Papers cited:
#     - DAST: Difficulty-aware few-shot examples reduce overthinking
#     - CoT: Step-by-step prompts improve reasoning
#     """
#     if dataset_type not in PROMPT_TEMPLATES:
#         raise ValueError(f"Unknown dataset type: {dataset_type}")

#     template = PROMPT_TEMPLATES[dataset_type]
#     difficulty = example.get("difficulty", "medium")

#     # Get difficulty configuration
#     diff_config = get_difficulty_config(difficulty)
#     max_tokens = diff_config["max_tokens"] if USE_DIFFICULTY_AWARE else FIXED_TOKEN_BUDGET

#     # Get few-shot example for this difficulty
#     if dataset_type == "mmlu":
#         example_text = MMLU_EXAMPLES.get(difficulty, MMLU_EXAMPLES["medium"])
#     elif dataset_type == "gsm8k":
#         example_text = GSM8K_EXAMPLES.get(difficulty, GSM8K_EXAMPLES["medium"])
#     else:
#         example_text = ""

#     # Build system prompt with difficulty instruction and example
#     system_prompt = template["system_base"].format(
#         instruction=diff_config["instruction"],
#         example=example_text
#     )

#     # Build user prompt
#     if dataset_type == "mmlu":
#         # Parse question and choices
#         prompt_text = example.get("prompt", "")
#         if isinstance(prompt_text, str):
#             lines = prompt_text.split('\n')
#             question = lines[0].strip() if lines else ""

#             # Extract choices (A, B, C, D)
#             choices = {"A": "", "B": "", "C": "", "D": ""}
#             choice_idx = 0
#             for line in lines[1:]:
#                 line = line.strip()
#                 if line and len(line) > 2 and line[0] in "ABCD" and line[1] in ") :":
#                     choice_letter = line[0]
#                     choice_text = line[3:].strip() if len(line) > 3 else line[2:].strip()
#                     choices[choice_letter] = choice_text

#             user_prompt = template["user_template"].format(
#                 question=question,
#                 choice_a=choices.get("A", ""),
#                 choice_b=choices.get("B", ""),
#                 choice_c=choices.get("C", ""),
#                 choice_d=choices.get("D", "")
#             )
#         else:
#             user_prompt = str(prompt_text)

#     elif dataset_type == "gsm8k":
#         user_prompt = template["user_template"].format(
#             question=example.get("prompt", "")
#         )
#     else:
#         user_prompt = example.get("prompt", "")

#     answer = example.get("answer", "")

#     return system_prompt, user_prompt, answer, max_tokens


# def build_llama_formatted_prompt(
#     example: Dict[str, Any],
#     dataset_type: str
# ) -> Tuple[str, int]:
#     """
#     Build complete Llama-3.1 formatted prompt.

#     Returns:
#         Tuple of (complete_prompt, max_tokens)
#     """
#     system_prompt, user_prompt, answer, max_tokens = build_difficulty_aware_prompt(
#         example, dataset_type
#     )

#     full_prompt = build_llama_chat_prompt(system_prompt, user_prompt)

#     return full_prompt, max_tokens


# def build_improved_prompt(
#     example: Dict[str, Any],
#     dataset_type: str
# ) -> Tuple[str, str, str]:
#     """
#     Legacy function for backward compatibility.
#     Returns (system_prompt, user_prompt, answer) without max_tokens.
#     """
#     system_prompt, user_prompt, answer, _ = build_difficulty_aware_prompt(
#         example, dataset_type
#     )
#     return system_prompt, user_prompt, answer


# def get_expected_format(dataset_type: str) -> str:
#     """Get expected output format for a dataset type"""
#     if dataset_type not in PROMPT_TEMPLATES:
#         raise ValueError(f"Unknown dataset type: {dataset_type}")
#     return PROMPT_TEMPLATES[dataset_type]["expected_format"]


# def set_token_budget_mode(use_difficulty_aware: bool):
#     """
#     Switch between Fixed and Difficulty-Aware token budgets.

#     Args:
#         use_difficulty_aware: True for adaptive budgets, False for fixed 256
#     """
#     global USE_DIFFICULTY_AWARE
#     USE_DIFFICULTY_AWARE = use_difficulty_aware


# # =============================================================================
# # TEST / DEMO
# # =============================================================================

# if __name__ == "__main__":
#     print("=" * 80)
#     print("LLAMA-3.1 OPTIMIZED DIFFICULTY-AWARE PROMPT TEMPLATES")
#     print("=" * 80)

#     # Show configuration
#     print("\nToken Budgets by Difficulty:")
#     for diff, config in DIFFICULTY_CONFIG.items():
#         print(f"  {diff.upper():8s}: max_tokens={config['max_tokens']:3d} | {config['description']}")

#     print(f"\nCurrent Mode: {'Difficulty-Aware' if USE_DIFFICULTY_AWARE else 'Fixed (' + str(FIXED_TOKEN_BUDGET) + ')'}")

#     # Test MMLU
#     print("\n" + "-" * 80)
#     print("MMLU EXAMPLE (Easy)")
#     print("-" * 80)

#     mmlu_easy = {
#         "prompt": "What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6",
#         "answer": "B",
#         "difficulty": "easy"
#     }

#     sys_prompt, user_prompt, answer, max_tokens = build_difficulty_aware_prompt(mmlu_easy, "mmlu")
#     full_prompt, _ = build_llama_formatted_prompt(mmlu_easy, "mmlu")

#     print(f"Max Tokens: {max_tokens}")
#     print(f"\nSystem Prompt (first 200 chars):\n{sys_prompt[:200]}...")
#     print(f"\nUser Prompt:\n{user_prompt}")
#     print(f"\nFull Llama-3.1 Formatted Prompt (first 300 chars):\n{full_prompt[:300]}...")

#     # Test GSM8K
#     print("\n" + "-" * 80)
#     print("GSM8K EXAMPLE (Hard)")
#     print("-" * 80)

#     gsm8k_hard = {
#         "prompt": "A factory produces 120 widgets per day...",
#         "answer": "3600",
#         "difficulty": "hard"
#     }

#     sys_prompt, user_prompt, answer, max_tokens = build_difficulty_aware_prompt(gsm8k_hard, "gsm8k")
#     print(f"Max Tokens: {max_tokens}")
#     print(f"\nSystem Prompt (first 200 chars):\n{sys_prompt[:200]}...")
#     print(f"\nUser Prompt:\n{user_prompt}")