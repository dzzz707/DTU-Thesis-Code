import os
import re
import ast
import json
from typing import Tuple, Dict, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class TestCategory(Enum):
    VALID_FORMULA = "Valid Formula"
    REQUIRES_CLEANING = "Requires Cleaning"
    INVALID_INPUT = "Invalid Input"

@dataclass
class TestCase:
    id: str
    input_formula: str
    expected_result: bool
    category: TestCategory
    description: str

class FormulaValidator:

    def _clean_formula(self, formula: str) -> str:
        # Stage 1: Formula Cleaning
        ending_words = [
            'justification', 'reason', 'because', 
            'to enhance', 'will improve', 'provide'
        ]
        formula_lower = formula.lower()
        
        for word in ending_words:
            if word in formula_lower:
                pos = formula_lower.find(word)
                formula = formula[:pos].strip()
                break
        
        formula = formula.strip('"\'{}[]').strip()
        formula = formula.rstrip('.,;:!')
        
        latex_conversions = {
            '\\times': '*',
            '\\text{': '',
            '}': '',
            '\\frac{': '(',
            '}{': ') / (',
            '\\': ''
        }
        
        for latex, python in latex_conversions.items():
            formula = formula.replace(latex, python)
        
        open_count = formula.count('(')
        close_count = formula.count(')')
        
        if open_count > close_count:
            formula += ')' * (open_count - close_count)
        elif close_count > open_count:
            formula = '(' * (close_count - open_count) + formula
        
        formula = re.sub(r'\s+', ' ', formula)
        
        return formula.strip()
    
    def _validate_formula_syntax(self, formula: str) -> bool:
        # Stage 2: AST-based Syntax Validation
        try:
            ast.parse(formula)
            
            stack = []
            for char in formula:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if not stack:
                        return False
                    stack.pop()
            
            return len(stack) == 0
            
        except SyntaxError:
            return False
    
    def _is_valid_formula(self, stub: str, formula: str) -> bool:
        # Stage 3: Semantic Validation
        if len(formula) < 3 or len(formula) > 500:
            return False
        
        descriptive_words = [
            'consider', 'should', 'include', 
            'improvement', 'enhance', 'recommend'
        ]
        if any(word in formula.lower() for word in descriptive_words):
            return False
        
        math_operators = ['+', '-', '*', '/', '(', ')']
        if not any(op in formula for op in math_operators):
            return False
        
        english_words = ['the', 'and', 'or', 'for', 'with', 'more', 'accurate']
        word_count = sum(1 for word in english_words if word in formula.lower())
        if word_count > 2:
            return False
        
        return True
    
    def validate(self, formula: str) -> Tuple[bool, str, str]:
        cleaned = self._clean_formula(formula)
        
        if not self._validate_formula_syntax(cleaned):
            return False, "SYNTAX_ERROR", cleaned
        
        if not self._is_valid_formula("test", cleaned):
            return False, "SEMANTIC_ERROR", cleaned
        
        return True, "PASSED", cleaned

TEST_CASES: List[TestCase] = [

    # Category A: Valid Formula (10 cases)
    # These represent correctly formatted LLM outputs following prompt guidelines
    TestCase(
        id="A01",
        input_formula="Fuel_Consumption * Emission_Factor",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Basic multiplication formula with underscored variable names"
    ),
    TestCase(
        id="A02",
        input_formula="(m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP)) / 1000",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Complex nested formula matching prompt example exactly"
    ),
    TestCase(
        id="A03",
        input_formula="(m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP) * EfficiencyFactor) / 1000",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Improved formula version from prompt example"
    ),
    TestCase(
        id="A04",
        input_formula="Energy_kWh * Grid_EF / 1000",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Scope 2 electricity emission calculation formula"
    ),
    TestCase(
        id="A05",
        input_formula="AirTravelDistance * AirTravel_CO2_Factor",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Scope 3 business travel formula from prompt"
    ),
    TestCase(
        id="A06",
        input_formula="Distance * Weight * EF_Transport",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Transportation emission formula with three factors"
    ),
    TestCase(
        id="A07",
        input_formula="Base * (1 + Rate / 100)",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Formula with percentage calculation"
    ),
    TestCase(
        id="A08",
        input_formula="(Scope1 + Scope2 + Scope3) * Correction_Factor",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Total emission aggregation formula"
    ),
    TestCase(
        id="A09",
        input_formula="Consumption * Factor * (1 - Loss_Rate)",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Formula with loss rate adjustment"
    ),
    TestCase(
        id="A10",
        input_formula="Activity_Data ** 0.5 * Coefficient",
        expected_result=True,
        category=TestCategory.VALID_FORMULA,
        description="Formula with power operation"
    ),
    
    
    # Category B: Requires Cleaning (10 cases)
    # These represent common LLM output deviations that cleaning should handle
    TestCase(
        id="B01",
        input_formula="A \\times B",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="LaTeX multiplication symbol (should convert to *)"
    ),
    TestCase(
        id="B02",
        input_formula="A \\times B \\times C",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Multiple LaTeX multiplication symbols"
    ),
    TestCase(
        id="B03",
        input_formula="Total \\times (1 + Rate)",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="LaTeX with parentheses (mixed format)"
    ),
    TestCase(
        id="B04",
        input_formula="A * B because it improves accuracy",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Valid formula with trailing justification text"
    ),
    TestCase(
        id="B05",
        input_formula="X + Y to enhance the calculation",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Formula with 'to enhance' trailing text"
    ),
    TestCase(
        id="B06",
        input_formula="Fuel * EF will improve accuracy",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Formula with 'will improve' trailing text"
    ),
    TestCase(
        id="B07",
        input_formula="  A  +  B  ",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Formula with excessive whitespace"
    ),
    TestCase(
        id="B08",
        input_formula="(A + B",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Unclosed parenthesis"
    ),
    TestCase(
        id="B09",
        input_formula="A + B)",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Extra closing parenthesis"
    ),
    TestCase(
        id="B10",
        input_formula="\"A * B\"",
        expected_result=True,
        category=TestCategory.REQUIRES_CLEANING,
        description="Formula wrapped in quotation marks"
    ),
    

    # Category C: Invalid Input (10 cases)
    # These represent LLM outputs that should be rejected by the validation
    TestCase(
        id="C01",
        input_formula="should include efficiency factor",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Descriptive text with 'should' keyword"
    ),
    TestCase(
        id="C02",
        input_formula="recommend using a better formula",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Descriptive text with 'recommend' keyword"
    ),
    TestCase(
        id="C03",
        input_formula="consider adding the correction factor",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Descriptive text with 'consider' keyword"
    ),
    TestCase(
        id="C04",
        input_formula="this is an improvement for accuracy",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Descriptive text with 'improvement' keyword"
    ),
    TestCase(
        id="C05",
        input_formula="enhance the calculation method",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Descriptive text with 'enhance' keyword"
    ),
    TestCase(
        id="C06",
        input_formula="AB",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Too short and no mathematical operator"
    ),
    TestCase(
        id="C07",
        input_formula="",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Empty string input"
    ),
    TestCase(
        id="C08",
        input_formula="the formula for the calculation with more accurate results",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="Excessive English words (>2 common words)"
    ),
    TestCase(
        id="C09",
        input_formula="No formula available",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="LLM failure response (no formula content)"
    ),
    TestCase(
        id="C10",
        input_formula="Please refer to the document for details",
        expected_result=False,
        category=TestCategory.INVALID_INPUT,
        description="LLM deflection response"
    ),
]

def run_validation_tests() -> Dict:

    print("\n" + "=" * 72)
    print("FORMULA INPUT VALIDATION TEST SUITE")
    print("=" * 72)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Test Cases: {len(TEST_CASES)}")
    print("=" * 72)
    
    validator = FormulaValidator()
    
    results = {
        TestCategory.VALID_FORMULA: [],
        TestCategory.REQUIRES_CLEANING: [],
        TestCategory.INVALID_INPUT: []
    }
    
    for case in TEST_CASES:
        passed, rejection_stage, cleaned = validator.validate(case.input_formula)
        is_correct = (passed == case.expected_result)
        
        result = {
            "id": case.id,
            "input": case.input_formula,
            "expected": case.expected_result,
            "actual": passed,
            "correct": is_correct,
            "rejection_stage": rejection_stage,
            "cleaned_formula": cleaned,
            "description": case.description
        }
        
        results[case.category].append(result)
    
    # Print detailed results by category
    category_info = [
        (TestCategory.VALID_FORMULA, "Category A: Valid Formula", "Expected: ACCEPT"),
        (TestCategory.REQUIRES_CLEANING, "Category B: Requires Cleaning", "Expected: ACCEPT after cleaning"),
        (TestCategory.INVALID_INPUT, "Category C: Invalid Input", "Expected: REJECT"),
    ]
    
    for category, title, expectation in category_info:
        cat_results = results[category]
        correct_count = sum(1 for r in cat_results if r["correct"])
        
        print(f"\n{title}")
        print(f"{expectation}")
        print(f"Result: {correct_count}/{len(cat_results)}")
        print("-" * 72)
        
        for r in cat_results:
            status = "PASS" if r["correct"] else "FAIL"
            input_display = r["input"][:45] + "..." if len(r["input"]) > 45 else r["input"]
            if not input_display:
                input_display = "(empty string)"
            
            print(f"  [{r['id']}] {status}")
            print(f"       Input: {input_display}")
            print(f"       Description: {r['description']}")
            
            if not r["correct"]:
                expected_str = "ACCEPT" if r["expected"] else "REJECT"
                actual_str = "ACCEPT" if r["actual"] else f"REJECT ({r['rejection_stage']})"
                print(f"       Expected: {expected_str}, Actual: {actual_str}")
            print()
    
    return results


def compute_statistics(results: Dict) -> Dict:
    stats = {}
    total_correct = 0
    total_cases = 0
    
    for category in TestCategory:
        cat_results = results[category]
        correct = sum(1 for r in cat_results if r["correct"])
        total = len(cat_results)
        accuracy = correct / total * 100 if total > 0 else 0
        
        stats[category.value] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        
        total_correct += correct
        total_cases += total
    
    stats["overall"] = {
        "correct": total_correct,
        "total": total_cases,
        "accuracy": total_correct / total_cases * 100 if total_cases > 0 else 0
    }
    
    return stats

if __name__ == "__main__":
    results = run_validation_tests()
    stats = compute_statistics(results)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "validation_outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"formula_validation_results_{timestamp}.json"
    full_output_path = os.path.join(output_dir, output_filename)
    
    json_results = {category.value: results[category] for category in TestCategory}
    
    output_data = {
        "test_execution_time": datetime.now().isoformat(),
        "total_test_cases": len(TEST_CASES),
        "statistics": stats,
        "detailed_results": json_results
    }
    
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to:{full_output_path}")