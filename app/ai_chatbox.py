import re
import streamlit as st
import json
import pandas as pd
from datetime import datetime
import uuid
import ast
from typing import List, Dict, Optional, Set
import PyPDF2
import io
import base64
from model_manager import get_model_scope, is_total_model

class ChatInterface:
    
    def __init__(self, api_key: str = None, manager=None):
        self.api_key = api_key
        self.client = None
        self.manager = manager
        if api_key:
            self.initialize_openai_client(api_key)
    
    def initialize_openai_client(self, api_key: str) -> None:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.api_key = api_key
        except ImportError:
            st.error("OpenAI package not installed. Please install with: pip install openai")
            self.client = None
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
    
    def get_all_model_variables(self) -> Dict[str, Dict]:
        variables = {}
        
        if not self.manager:
            return variables
        
        for model_name, model in self.manager.models.items():
            for stub, mode in model.data_modes.items():
                variables[stub] = {
                    "type": "parameter",
                    "name": mode.name,
                    "value": mode.value,
                    "model": model_name,
                    "stub": stub,
                    "keywords": self.generate_keywords(mode.name, stub)
                }
            
            for stub, mode in model.formula_modes.items():
                variables[stub] = {
                    "type": "formula",
                    "name": mode.name,
                    "formula": mode.formula,
                    "result": mode.result if hasattr(mode, 'result') else None,
                    "model": model_name,
                    "stub": stub,
                    "keywords": self.generate_keywords(mode.name, stub)
                }
        
        return variables
    
    def get_emission_factors_only(self) -> Dict[str, Dict]:
        factors = {}
        
        if not self.manager:
            return factors
        
        for model_name, model in self.manager.models.items():
            for stub, mode in model.data_modes.items():
                if self.is_emission_factor(mode.name, stub):
                    factors[stub] = {
                        "type": "emission_factor",
                        "name": mode.name,
                        "value": mode.value,
                        "model": model_name,
                        "stub": stub,
                        "keywords": self.generate_keywords(mode.name, stub)
                    }
        
        return factors
    
    def is_emission_factor(self, name: str, stub: str) -> bool:
        name_lower = name.lower()
        stub_lower = stub.lower()
        
        factor_keywords = [
            'factor', 'emission factor', 'gwp', 'xgwp', 
            'co2', 'ch4', 'n2o', 'coefficient', 'ratio'
        ]
        
        for keyword in factor_keywords:
            if keyword in name_lower:
                return True
        
        for keyword in ['gwp', 'xgwp', 'co2', 'ch4', 'n2o']:
            if keyword in stub_lower:
                return True
        
        if any(stub_lower.endswith(pattern) for pattern in ['_gwp', '_xgwp', '_co2', '_ch4', '_n2o']):
            return True
        
        return False
    
    def get_current_formulas(self) -> Dict[str, Dict]:
        formulas = {}
        
        if not self.manager:
            return formulas
        
        for model_name, model in self.manager.models.items():
            for stub, mode in model.formula_modes.items():
                formulas[stub] = {
                    "name": mode.name,
                    "formula": mode.formula,
                    "model": model_name,
                    "result": mode.result if hasattr(mode, 'result') else None
                }
        
        return formulas
    
    def get_emission_sources_structure(self) -> Dict[str, List]:
        sources = {
            "Scope 1": [],
            "Scope 2": [],
            "Scope 3": []
        }
        
        if not self.manager:
            return sources
        
        # Dynamically detect scope based on model name prefix
        for model_name, model in self.manager.models.items():
            scope_key = get_model_scope(model_name)
            
            # Map internal scope key to display name
            scope_display_map = {
                "Scope1": "Scope 1",
                "Scope2": "Scope 2",
                "Scope3": "Scope 3"
            }
            
            scope = scope_display_map.get(scope_key, "Other")
            
            if scope in sources:
                for stub, mode in model.formula_modes.items():
                    # Skip total models and non-CO2e results
                    if is_total_model(model_name):
                        continue
                    if not stub.endswith("CO2e"):
                        continue
                    
                    sources[scope].append({
                        "name": mode.name,
                        "stub": stub,
                        "model": model_name,
                        "formula": mode.formula
                    })
        
        return sources
    
    def generate_keywords(self, name: str, stub: str) -> List[str]:
        keywords = []
        
        keywords.append(name.lower())
        keywords.append(stub.lower())
        
        name_words = re.findall(r'\w+', name.lower())
        keywords.extend(name_words)
        
        stub_parts = re.sub(r'([A-Z])', r' \1', stub).split()
        keywords.extend([p.lower() for p in stub_parts])
        
        if 'co2' in stub.lower():
            keywords.extend(['carbon dioxide', 'carbon'])
        if 'ch4' in stub.lower():
            keywords.extend(['methane'])
        if 'n2o' in stub.lower():
            keywords.extend(['nitrous oxide'])
        if 'gwp' in stub.lower() or 'xgwp' in stub.lower():
            keywords.extend(['global warming potential', 'emission factor', 'factor'])
        if 'electricity' in name.lower():
            keywords.extend(['power', 'energy', 'kwh', 'mwh'])
        if 'gas' in name.lower():
            keywords.extend(['natural gas', 'lng', 'fuel'])
        if 'diesel' in name.lower():
            keywords.extend(['fuel', 'oil', 'diesel fuel'])
        
        return list(set(keywords))
    
    def create_variable_search_context(self) -> str:
        factors = self.get_emission_factors_only()
        
        if not factors:
            return "No emission factors found in the system."
        
        context = "EMISSION FACTORS TO SEARCH FOR IN DOCUMENTS:\n\n"
        
        models_dict = {}
        for stub, info in factors.items():
            model = info['model']
            if model not in models_dict:
                models_dict[model] = []
            models_dict[model].append((stub, info))
        
        for model_name, vars_list in models_dict.items():
            context += f"\n{model_name}:\n"
            for stub, info in vars_list:
                context += f"  - {stub}: {info['name']}\n"
                context += f"    Current Value: {info['value']}\n"
                context += f"    Search Keywords: {', '.join(info['keywords'][:5])}\n"
        
        context += f"\nNOTE: Only search for EMISSION FACTORS (factors, coefficients, GWP values). \n"
        context += f"Do NOT match activity data like consumption amounts, usage quantities, etc.\n"
        
        return context
    
    def create_formula_analysis_context(self) -> str:
        formulas = self.get_current_formulas()
        
        if not formulas:
            return "No formulas found in the system."
        
        context = "CURRENT CALCULATION FORMULAS FOR ANALYSIS:\n\n"
        
        for stub, info in formulas.items():
            context += f"{stub} ({info['name']}):\n"
            context += f"  Formula: {info['formula']}\n"
            context += f"  Model: {info['model']}\n"
            if info['result'] is not None:
                context += f"  Current Result: {info['result']:.6f}\n"
            context += "\n"
        
        return context
    
    def create_gap_analysis_context(self) -> str:
        sources = self.get_emission_sources_structure()
        
        context = "CURRENT EMISSION SOURCES BY SCOPE:\n\n"
        
        for scope, source_list in sources.items():
            context += f"{scope}:\n"
            if source_list:
                for source in source_list:
                    context += f"  - {source['name']} ({source['stub']})\n"
            else:
                context += "  - No emission sources found\n"
            context += "\n"
        
        context += "\nCOMMON EMISSION SOURCES TO LOOK FOR:\n"
        context += "Scope 1: Natural gas, diesel fuel, gasoline, LPG, refrigerants, process emissions\n"
        context += "Scope 2: Purchased electricity, steam, heating, cooling\n"
        context += "Scope 3: Business travel, waste, water, paper, commuting, supply chain\n"
        
        return context
    
    def analyze_formula_improvements(self) -> str:
        """Analyze current formulas and suggest improvements"""
        formula_context = self.create_formula_analysis_context()
        
        improvement_prompt = f"""As a carbon emission calculation expert, analyze our current calculation formulas and provide specific improvement recommendations.

{formula_context}

COMPREHENSIVE FORMULA ANALYSIS:

1. ACCURACY ASSESSMENT: Evaluate each formula for calculation accuracy and completeness
2. METHODOLOGY REVIEW: Compare against industry best practices and standards
3. FACTOR COMPLETENESS: Identify missing emission factors or variables
4. PRECISION OPPORTUNITIES: Suggest more precise calculation methods
5. STANDARDIZATION: Recommend alignment with recognized emission calculation standards

ANALYSIS STRUCTURE:
For each significant finding, provide:
- Formula Name and Current Method
- Identified Issues or Limitations  
- Specific Improvement Recommendations
- Justification for Changes
- Implementation Priority (High/Medium/Low)

==============================================================================
CRITICAL OUTPUT FORMAT REQUIREMENT - YOU MUST FOLLOW THIS EXACTLY:
==============================================================================

For EACH formula improvement, you MUST output in this EXACT format with NO variations:

FORMULA IMPROVEMENT: [exact_stub_name]
Current: [current_formula]
Improved: [new_formula_in_python_syntax]
Justification: [reason for change]

RULES YOU MUST FOLLOW:
1. The line MUST start with exactly "FORMULA IMPROVEMENT: " (with colon and space)
2. Do NOT use numbered lists like "1. NaturalGasCO2e" - this will cause parsing failure
3. Do NOT use bullet points or dashes before the stub name
4. Do NOT add blank lines between FORMULA IMPROVEMENT and Current
5. Each field (Current, Improved, Justification) must be on its own line immediately after the previous one

CORRECT EXAMPLE:
FORMULA IMPROVEMENT: NaturalGasCO2e
Current: (m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP)) / 1000
Improved: (m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP) * EfficiencyFactor) / 1000
Justification: Include combustion efficiency factor for more accurate calculations

WRONG EXAMPLES (DO NOT USE THESE FORMATS):
1. NaturalGasCO2e
- FORMULA IMPROVEMENT: NaturalGasCO2e
**FORMULA IMPROVEMENT**: NaturalGasCO2e
Formula Improvement: NaturalGasCO2e (wrong capitalization)

==============================================================================

Focus on practical improvements that would enhance calculation accuracy, completeness, and compliance with emission reporting standards."""

        if not self.client:
            return "Error: OpenAI client not initialized. Please check your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing formulas: {str(e)}"

    def analyze_emission_gaps(self) -> str:
        gap_context = self.create_gap_analysis_context()
        
        gap_analysis_prompt = f"""As an emission inventory expert, conduct a comprehensive gap analysis of our current emission model to identify missing emission sources and coverage gaps.

{gap_context}

COMPREHENSIVE GAP ANALYSIS:

1. SCOPE COVERAGE ASSESSMENT:
- Scope 1: Direct emission sources we may be missing
- Scope 2: Indirect energy-related emissions gaps  
- Scope 3: Value chain emissions not currently captured

2. INDUSTRY STANDARDS COMPARISON:
- Compare against GHG Protocol requirements
- Identify ISO 14064 compliance gaps
- Check against sector-specific emission guidelines

3. COMMON EMISSION SOURCES REVIEW:
- Transportation: Fleet vehicles, business travel, employee commuting
- Energy: Electricity, heating, cooling, steam
- Operations: Process emissions, fugitive emissions, waste
- Supply Chain: Purchased goods, upstream/downstream activities

4. REGULATORY COMPLIANCE:
- Mandatory reporting requirements we may be missing
- Voluntary standards we should consider

==============================================================================
CRITICAL OUTPUT FORMAT REQUIREMENT - YOU MUST FOLLOW THIS EXACTLY:
==============================================================================

For EACH new emission source, you MUST output in this EXACT format with NO variations:

NEW SOURCE: [source name]
Variable: [suggested_stub_name]
Formula: [calculation_formula]
Scope: [Scope 1/2/3]
Priority: [High/Medium/Low]
Justification: [why needed]

RULES YOU MUST FOLLOW:
1. The line MUST start with exactly "NEW SOURCE: " (with colon and space)
2. The next line MUST start with exactly "Variable: " (NO blank lines between them)
3. Do NOT use numbered lists like "1. NEW SOURCE:" - this will cause parsing failure
4. Do NOT add blank lines between any of the fields
5. All 6 fields must be present and on consecutive lines

CORRECT EXAMPLE:
NEW SOURCE: Business Air Travel
Variable: BusinessAirTravelCO2e
Formula: AirTravelDistance * AirTravel_CO2_Factor
Scope: Scope 3
Priority: High
Justification: Business travel is a significant Scope 3 emission source required by GHG Protocol

WRONG EXAMPLES (DO NOT USE THESE FORMATS):
1. NEW SOURCE: Business Air Travel
NEW SOURCE: Business Air Travel
   
   Variable: BusinessAirTravelCO2e  (blank line before Variable - WRONG)
- NEW SOURCE: Business Air Travel
**NEW SOURCE**: Business Air Travel

==============================================================================

You MUST provide at least 3-5 new sources using the EXACT format above.

ANALYSIS OUTPUT:
For each identified gap:
- Missing Emission Source Category
- Applicable Scope Classification
- Typical Calculation Methodology
- Required Data Collection
- Implementation Priority and Complexity
- Potential Impact on Total Emissions

Provide specific, actionable recommendations for improving our emission inventory completeness."""

        if not self.client:
            return "Error: OpenAI client not initialized. Please check your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": gap_analysis_prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing gaps: {str(e)}"

    def analyze_model_structure(self) -> str:
        structure = self.get_emission_sources_structure()
        formulas = self.get_current_formulas()
        variables = self.get_all_model_variables()
        
        analysis_prompt = f"""As a carbon accounting expert, provide an analytical overview of our current emission calculation model.

MODEL STATISTICS:
- Total Variables: {len(variables)}
- Total Formulas: {len(formulas)}
- Scope 1 Sources: {len(structure.get('Scope 1', []))}
- Scope 2 Sources: {len(structure.get('Scope 2', []))}
- Scope 3 Sources: {len(structure.get('Scope 3', []))}

CURRENT STRUCTURE:
{json.dumps(structure, indent=2)}

CURRENT FORMULAS:
{json.dumps({stub: info["formula"] for stub, info in formulas.items()}, indent=2)}

Please provide:
1. Overview of the model structure and organization
2. Assessment of scope coverage
3. Summary of calculation methodologies used
4. Observations about data completeness
5. General compliance status with GHG Protocol

Focus on describing and assessing the current state, not on making improvement recommendations."""

        if not self.client:
            return "Error: OpenAI client not initialized. Please check your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.5,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing model: {str(e)}"


    def check_standards_compliance(self) -> str:
        """Check model compliance with major carbon accounting standards"""
        structure = self.get_emission_sources_structure()
        formulas = self.get_current_formulas()
        
        compliance_prompt = f"""As a carbon accounting standards expert, evaluate our emission calculation model against major international standards including ISO 14064-1, GHG Protocol, and IPCC Guidelines.

    CURRENT MODEL STRUCTURE:
    {json.dumps(structure, indent=2)}

    CURRENT FORMULAS:
    {json.dumps({stub: info["formula"] for stub, info in formulas.items()}, indent=2)}

    Please evaluate compliance with the following standards:

    1. ISO 14064-1 COMPLIANCE
    - Organizational and operational boundary definition
    - Scope 1/2/3 emission categorization
    - Quantification methodology alignment

    2. GHG PROTOCOL ALIGNMENT
    - Corporate Standard requirements
    - Scope 2 Guidance (location-based vs market-based)
    - Scope 3 Standard categories coverage

    3. IPCC GUIDELINES CONSISTENCY
    - Emission factor tier approach (Tier 1/2/3)
    - Activity data quality requirements
    - Good practice guidance adherence

    4. CROSS-STANDARD ASSESSMENT
    - Completeness across all frameworks
    - Consistency in calculation approaches
    - Transparency and documentation requirements
    - Uncertainty considerations

    Provide a compliance assessment with specific references to relevant clauses and sections from each standard."""

        if not self.client:
            return "Error: OpenAI client not initialized. Please check your API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": compliance_prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error checking compliance: {str(e)}"

    # Specialized Prompt Templates
    def get_parameter_matching_prompt(self) -> str:
        """System prompt for emission factor matching only"""
        variable_context = self.create_variable_search_context()
        
        return f"""You are a carbon emission factor extraction assistant. Your primary task is to find and match EMISSION FACTORS from the document content provided below to the existing emission factors in our system.

IMPORTANT: The document content has been extracted and included in the user's message. You CAN and SHOULD read and analyze this content directly.

{variable_context}

CRITICAL INSTRUCTIONS:
1. ONLY search for EMISSION FACTORS (coefficients, GWP values, factors)
2. DO NOT match activity data (consumption amounts, usage quantities, distance traveled, etc.)
3. Look for values that represent emissions per unit of activity
4. Use the keywords provided to identify relevant emission factors
5. Match document values to existing emission factors whenever possible

EMISSION FACTORS include:
- Global Warming Potential (GWP) values
- CO2 emission coefficients
- CH4 emission factors
- N2O emission factors
- Conversion factors for different fuels
- Electricity grid emission factors

ACTIVITY DATA to IGNORE:
- Fuel consumption amounts (liters, gallons, m3)
- Electricity usage (kWh, MWh)
- Distance traveled (km, miles)
- Quantities of materials used
- Number of employees or vehicles

==============================================================================
CRITICAL OUTPUT FORMAT REQUIREMENT - YOU MUST FOLLOW THIS EXACTLY:
==============================================================================

For EACH emission factor match, you MUST output in this EXACT format with NO variations:

FACTOR MATCH: [exact_stub_name] - [factor name]
Current Factor: [current_value]
Document Factor: [new_value] [unit]
Location: [page/section reference]

RULES YOU MUST FOLLOW:
1. The line MUST start with exactly "FACTOR MATCH: " (with colon and space)
2. Do NOT use numbered lists like "1. FACTOR MATCH:" - this will cause parsing failure
3. Do NOT use bullet points or dashes before the stub name
4. Do NOT add blank lines between any of the four fields
5. All 4 fields must be present and on consecutive lines
6. Use the EXACT stub name from the emission factors list above

CORRECT EXAMPLE:
FACTOR MATCH: CO2_XGWP - CO2 Emission Factor
Current Factor: 1.87904
Document Factor: 1.92 kg CO2/m3
Location: Page 15, Table 3.2

WRONG EXAMPLES (DO NOT USE THESE FORMATS):
1. FACTOR MATCH: CO2_XGWP
- FACTOR MATCH: CO2_XGWP
**FACTOR MATCH**: CO2_XGWP
FACTOR MATCH: CO2_XGWP - CO2 Emission Factor

   Current Factor: 1.87904  (blank line between fields - WRONG)

==============================================================================

IF NO EMISSION FACTORS ARE FOUND IN THE DOCUMENT:
If the document does not contain any specific numerical emission factor values (like GWP values, CO2 coefficients, etc.), you should:
1. Clearly state: "NO EMISSION FACTORS FOUND in this document."
2. Explain why: The document may be a framework/standard (like ISO 14064) that describes methodology but doesn't contain actual factor values.
3. Suggest alternative sources where the user could find actual emission factors (e.g., IPCC reports, EPA databases, national emission factor databases).
4. Briefly describe what the document DOES contain that might be useful.

Do NOT pretend to find factors if none exist. Be honest about the document's contents.
"""
    
    def get_formula_suggestion_prompt(self) -> str:
        """System prompt for formula suggestions with structured output format"""
        formula_context = self.create_formula_analysis_context()
        
        return f"""You are a carbon emission calculation expert analyzing our current calculation methods. Based on the document content provided, provide comprehensive formula improvement suggestions.

IMPORTANT: The document content has been extracted and included in the user's message. You CAN and SHOULD read and analyze this content directly.

{formula_context}

ANALYSIS INSTRUCTIONS:
1. Review our current calculation formulas and identify areas for improvement
2. Look for more accurate calculation methodologies in the document
3. Suggest enhanced formulas that could improve calculation precision
4. Identify calculation best practices we should adopt
5. Recommend additional variables or factors that could enhance accuracy

==============================================================================
CRITICAL OUTPUT FORMAT REQUIREMENT - YOU MUST FOLLOW THIS EXACTLY:
==============================================================================

For EACH formula improvement, you MUST output in this EXACT format with NO variations:

FORMULA IMPROVEMENT: [exact_stub_name]
Current: [current_formula]
Improved: [new_formula_in_python_syntax]
Justification: [reason for change]

RULES YOU MUST FOLLOW:
1. The line MUST start with exactly "FORMULA IMPROVEMENT: " (with colon and space)
2. Do NOT use numbered lists like "1. NaturalGasCO2e" - this will cause parsing failure
3. Do NOT use bullet points or dashes before the stub name
4. Do NOT add blank lines between FORMULA IMPROVEMENT and Current
5. Each field (Current, Improved, Justification) must be on its own line immediately after the previous one

CORRECT EXAMPLE:
FORMULA IMPROVEMENT: NaturalGasCO2e
Current: (m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP)) / 1000
Improved: (m3 * (CH4_XGWP + CO2_XGWP + N2O_XGWP) * EfficiencyFactor) / 1000
Justification: Include combustion efficiency factor for more accurate calculations

WRONG EXAMPLES (DO NOT USE THESE FORMATS):
1. NaturalGasCO2e
- FORMULA IMPROVEMENT: NaturalGasCO2e
**FORMULA IMPROVEMENT**: NaturalGasCO2e

==============================================================================

IMPORTANT RULES:
- Use exact stub names from the formulas list above
- Write improved formulas in Python syntax (use *, /, +, -, parentheses)
- Do NOT use LaTeX notation in the improved formula
- Each formula must be on a single line
- Do NOT include descriptive text in the formula itself
- Variables must match existing parameter names in the system

After providing the structured improvements, you may add additional analysis and recommendations."""

    def get_gap_analysis_prompt(self) -> str:
        """System prompt for gap analysis"""
        gap_context = self.create_gap_analysis_context()
        
        return f"""You are an emission inventory expert conducting a comprehensive gap analysis of our current emission model against industry standards and best practices.

IMPORTANT: The document content has been extracted and included in the user's message. You CAN and SHOULD read and analyze this content directly.

{gap_context}

ANALYSIS INSTRUCTIONS:
1. Systematically compare our current emission sources with those mentioned in the document
2. Identify missing emission sources across all three scopes
3. Assess completeness of our emission inventory
4. Highlight industry-specific sources we may have overlooked
5. Evaluate regulatory compliance gaps

COMPREHENSIVE ANALYSIS FOCUS:
- Scope 1: Direct emissions we might be missing
- Scope 2: Indirect energy-related emissions gaps
- Scope 3: Value chain emissions not currently captured
- Industry-specific emission sources
- Emerging emission categories
- Regulatory requirements and standards compliance

==============================================================================
CRITICAL OUTPUT FORMAT REQUIREMENT - YOU MUST FOLLOW THIS EXACTLY:
==============================================================================

For EACH new emission source, you MUST output in this EXACT format with NO variations:

NEW SOURCE: [source name]
Variable: [suggested_stub_name]
Formula: [calculation_formula]
Scope: [Scope 1/2/3]
Priority: [High/Medium/Low]
Justification: [why needed]

RULES YOU MUST FOLLOW:
1. The line MUST start with exactly "NEW SOURCE: " (with colon and space)
2. The next line MUST start with exactly "Variable: " (NO blank lines between them)
3. Do NOT use numbered lists like "1. NEW SOURCE:" - this will cause parsing failure
4. Do NOT add blank lines between any of the fields
5. All 6 fields must be present and on consecutive lines

CORRECT EXAMPLE:
NEW SOURCE: Business Air Travel
Variable: BusinessAirTravelCO2e
Formula: AirTravelDistance * AirTravel_CO2_Factor
Scope: Scope 3
Priority: High
Justification: Business travel is a significant Scope 3 emission source required by GHG Protocol

WRONG EXAMPLES (DO NOT USE THESE FORMATS):
1. NEW SOURCE: Business Air Travel
NEW SOURCE: Business Air Travel
   
   Variable: BusinessAirTravelCO2e  (blank line before Variable - WRONG)
- NEW SOURCE: Business Air Travel

==============================================================================

You MUST provide at least 3-5 new sources using the EXACT format above.

Provide a thorough gap analysis that identifies what we're missing and explains why these additions would improve our emission inventory's comprehensiveness and accuracy.
"""
 
    def send_to_gpt(self, messages: List[Dict[str, str]], system_prompt: str = None) -> Optional[str]:
        if not self.client:
            return "Error: OpenAI client not initialized. Please check your API key."
        
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=full_messages,
                temperature=0.2,
                max_tokens=3000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error communicating with GPT: {str(e)}"

    def chat(self, user_message: str, conversation_history: List[Dict] = None) -> Optional[str]:
        if not self.client:
            return None
        
        structure = self.get_emission_sources_structure()
        variables = self.get_all_model_variables()
        
        system_prompt = f"""You are an AI assistant specializing in ISO 14064-1 carbon emission calculations.

CURRENT MODEL CONTEXT:
- Total Variables: {len(variables)}
- Emission Sources: {sum(len(sources) for sources in structure.values())}
- Scopes: Scope 1, Scope 2, Scope 3

You can:
1. Answer questions about emission calculations
2. Explain variables and formulas
3. Provide ISO 14064-1 guidance
4. Help with data interpretation

Be helpful, accurate, and reference the model context when relevant."""

        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history[-5:])
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Chat failed: {str(e)}")
            return None
    
    def generate_emission_factor_json(self, gpt_response: str) -> Optional[Dict]:
        json_data = self.export_current_parameters_json()
        changes_made = False
        
        match_pattern = r'FACTOR MATCH:\s*(\w+)\s*-\s*([^\n]+)[\n\s]*Current Factor:\s*([\d.]+)[\n\s]*Document Factor:\s*([\d.]+)'
        
        for match in re.finditer(match_pattern, gpt_response, re.IGNORECASE):
            stub = match.group(1)
            new_value = float(match.group(4))
            
            for model_name, model in self.manager.models.items():
                if stub in model.data_modes:
                    if model_name in json_data:
                        json_data[model_name][stub] = new_value
                        changes_made = True
                    break
        
        return json_data if changes_made else None

    def generate_formula_improvement_json(self, gpt_response: str) -> Optional[Dict]:
        json_data = self.export_current_formulas_json()
        changes_made = False
        new_formulas = {}  # For formulas that don't match existing stubs
        
        # Strict pattern: only match "FORMULA IMPROVEMENT: StubName" format
        pattern = r'FORMULA IMPROVEMENT:\s*(\w+)\s*\nCurrent:\s*[^\n]+\nImproved:\s*([^\n]+)'
        
        matches = list(re.finditer(pattern, gpt_response, re.IGNORECASE))
        matched_stubs = set()  # Avoid duplicates
        
        for match in matches:
            stub = match.group(1).strip()
            new_formula = match.group(2).strip()
            
            # Skip if already matched
            if stub in matched_stubs:
                continue
            
            new_formula = self._clean_formula(new_formula)
            
            if not self._validate_formula_syntax(new_formula):
                continue
            
            if self._is_valid_formula(stub, new_formula):
                matched_stubs.add(stub)
                
                # Try to find existing formula
                found_existing = False
                for model_name, model in self.manager.models.items():
                    if stub in model.formula_modes:
                        if model_name in json_data:
                            json_data[model_name][stub] = new_formula
                            changes_made = True
                            found_existing = True
                        break
                
                # If not found in existing formulas, add to new_formulas
                if not found_existing:
                    new_formulas[stub] = new_formula
                    changes_made = True
        
        # Add new formulas section if there are any
        if new_formulas:
            json_data["_new_formulas"] = new_formulas
        
        return json_data if changes_made else None

    def generate_emission_source_json(self, gpt_response: str) -> Optional[Dict]:
        json_data = {}
        
        # Strict pattern: only match "NEW SOURCE: X\nVariable: Y\nFormula: Z" format
        pattern = r'NEW SOURCE:\s*([^\n]+)\nVariable:\s*(\w+)\nFormula:\s*([^\n]+)'
        
        found_sources = []
        
        for match in re.finditer(pattern, gpt_response, re.IGNORECASE):
            source_name = match.group(1).strip()
            variable_stub = match.group(2).strip()
            formula = match.group(3).strip()
            
            # Clean up formula
            formula = re.sub(r'\s+', ' ', formula)
            formula = formula.rstrip('.,;:')
            
            # Avoid duplicates
            if variable_stub not in [s['variable_stub'] for s in found_sources]:
                found_sources.append({
                    'source_name': source_name,
                    'variable_stub': variable_stub,
                    'formula': formula
                })
        
        # Process found sources
        for source in found_sources:
            source_name = source['source_name']
            variable_stub = source['variable_stub']
            formula = source['formula']
            
            # Determine model based on source name using new naming convention
            # Default to Scope3_Misc for unmatched sources
            model_name = "Scope3_Misc"
            
            source_name_lower = source_name.lower()
            
            # Scope 1: Direct emissions
            if any(keyword in source_name_lower for keyword in ['fixed', 'natural gas', 'stationary', 'boiler', 'furnace', 'combustion']):
                model_name = "Scope1_FixedCombustion"
            elif any(keyword in source_name_lower for keyword in ['fugitive', 'refrigerant', 'leak', 'hvac', 'cooling', 'air conditioning']):
                model_name = "Scope1_Fugitive"
            elif any(keyword in source_name_lower for keyword in ['mobile', 'vehicle', 'transport', 'fleet', 'car', 'truck', 'diesel', 'petrol', 'gasoline']):
                model_name = "Scope1_Mobile"
            # Scope 2: Indirect energy
            elif any(keyword in source_name_lower for keyword in ['electricity', 'power', 'grid', 'energy']):
                model_name = "Scope2_Electricity"
            # Scope 3: Other indirect (default already set)
            elif any(keyword in source_name_lower for keyword in ['travel', 'business travel', 'air travel', 'commut']):
                model_name = "Scope3_Travel"
            elif any(keyword in source_name_lower for keyword in ['waste', 'disposal', 'landfill']):
                model_name = "Scope3_Waste"
            elif any(keyword in source_name_lower for keyword in ['water', 'sewage']):
                model_name = "Scope3_Water"
            elif any(keyword in source_name_lower for keyword in ['supply chain', 'purchased', 'goods', 'services']):
                model_name = "Scope3_SupplyChain"
            
            if model_name not in json_data:
                json_data[model_name] = []
            
            # Extract parameters from formula
            parameters = []
            vars_in_formula = self.extract_variables(formula)
            for var in vars_in_formula:
                # Check if variable already exists in any model
                var_exists = False
                for m in self.manager.models.values():
                    if var in m.data_modes or var in m.formula_modes:
                        var_exists = True
                        break
                
                if not var_exists:
                    parameters.append({
                        "name": var.replace('_', ' ').title(),
                        "stub": var,
                        "value": 0.0
                    })
            
            json_data[model_name].append({
                "source_name": source_name,
                "variable_stub": variable_stub,
                "formula": formula,
                "parameters": parameters
            })
        
        return json_data if json_data else None
    
    def export_current_parameters_json(self) -> Dict:
        export_data = {}
        
        if not self.manager:
            return export_data
        
        for model_name, model in self.manager.models.items():
            if model.data_modes:
                export_data[model_name] = {}
                for stub, mode in model.data_modes.items():
                    export_data[model_name][stub] = mode.value
        
        return export_data
    
    def export_current_formulas_json(self) -> Dict:
        export_data = {}
        
        if not self.manager:
            return export_data
        
        for model_name, model in self.manager.models.items():
            if model.formula_modes:
                export_data[model_name] = {}
                for stub, mode in model.formula_modes.items():
                    export_data[model_name][stub] = mode.formula
        
        return export_data
    
    def _clean_formula(self, formula: str) -> str:
        ending_words = ['justification', 'reason', 'because', 'to enhance', 'will improve', 'provide']
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
        if len(formula) < 3 or len(formula) > 500:
            return False
        
        descriptive_words = ['consider', 'should', 'include', 'improvement', 'enhance', 'recommend']
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

    def extract_variables(self, formula: str) -> Set[str]:
        try:
            return {node.id for node in ast.walk(ast.parse(formula)) if isinstance(node, ast.Name)}
        except:
            return set()

    @staticmethod
    def extract_chat_title(message_content: str) -> str:
        if not message_content or len(message_content.strip()) == 0:
            return "New Chat"
        
        content = message_content.strip()
        
        if "[File:" in content:
            if "\n\n[File:" in content:
                content = content.split("\n\n[File:")[0]
            elif content.startswith("[File:"):
                return "Document Analysis"
        
        content = re.sub(r'\s+', ' ', content).strip()
        
        if len(content) > 45:
            content = content[:42] + "..."
        
        return content if content else "New Chat"
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def build_file_context(self, uploaded_file) -> str:
        prompt = f"\n\n--- DOCUMENT CONTENT START ---\nFile: {uploaded_file.name}\n\n"
        
        try:
            if uploaded_file.type == "application/pdf":
                pdf_text = self.extract_text_from_pdf(uploaded_file)
                prompt += pdf_text[:8000]
                
            elif uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode("utf-8")
                prompt += content[:8000]
                
            elif uploaded_file.type == "application/json":
                content = json.loads(uploaded_file.read())
                prompt += json.dumps(content, indent=2)[:8000]
                
            elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                prompt += f"Columns: {list(df.columns)}\n"
                prompt += f"Data:\n{df.to_string()[:8000]}"
                
        except Exception as e:
            prompt += f"Error reading file: {str(e)}"
        
        prompt += "\n--- DOCUMENT CONTENT END ---\n\nPlease analyze the document content above."
        
        return prompt
    
    def parse_extraction_response(self, response: str) -> Optional[Dict]:
        extracted = {"matched": [], "new": []}
        
        try:
            match_pattern = r'FACTOR MATCH:\s*(\w+)\s*-\s*([^\n]+)[\n\s]*Current Factor:\s*([\d.]+)[\n\s]*Document Factor:\s*([\d.]+)'
            for match in re.finditer(match_pattern, response, re.IGNORECASE):
                extracted["matched"].append({
                    "stub": match.group(1),
                    "name": match.group(2).strip(),
                    "current_value": match.group(3),
                    "new_value": match.group(4)
                })
            
            new_pattern = r'NEW EMISSION FACTOR:\s*([^\n]+)[\n\s]*Factor Value:\s*([\d.]+)'
            for match in re.finditer(new_pattern, response, re.IGNORECASE):
                extracted["new"].append({
                    "name": match.group(1).strip(),
                    "value": match.group(2)
                })
            
            return extracted if (extracted["matched"] or extracted["new"]) else None
            
        except Exception as e:
            return None
    
    def add_chat_specific_styles(self):
        st.markdown("""
        <style>
        .chat-history-container {
            max-height: 65vh;
            overflow-y: auto;
            padding-right: 8px;
            margin-bottom: 1rem;
        }
        
        .file-attachment {
            background-color: #1e2329;
            border: 1px solid #3b4252;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            display: inline-block;
        }
        
        .welcome-container {
            text-align: center;
            padding: 2rem;
            margin: 2rem 0;
        }
        
        .welcome-container h2 {
            color: #4a90e2;
            margin-bottom: 1rem;
        }
        
        .stButton button {
            height: 60px !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            font-size: 0.9rem !important;
            line-height: 1.2 !important;
            padding: 8px 12px !important;
            text-align: center !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        .stFileUploader {
            background-color: #1a202c;
            border: 2px dashed #4a5568;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
        }
        
        .stFileUploader:hover {
            border-color: #4a90e2;
            background-color: #2d3748;
        }
        
        .update-card {
            background-color: #2d3748;
            border-left: 4px solid #4a90e2;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
        }
        
        .download-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background-color: #1f4e79;
            color: white !important;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            margin: 8px 0;
            transition: background-color 0.3s ease;
        }
        
        .download-link:hover {
            background-color: #2d5aa0;
            text-decoration: none;
            color: white !important;
        }
        
        .download-section {
            background-color: #2d3748;
            border: 1px solid #4a5568;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .download-section h4 {
            color: #4a90e2;
            margin-bottom: 1rem;
        }
        
        .upload-instruction {
            background-color: #1a365d;
            border-left: 4px solid #4a90e2;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        
        /* Chat History Panel Styles */
        .chat-history-panel {
            background-color: #1e1e2e;
            border-right: 1px solid #3a3a4a;
            height: 100%;
            overflow-y: auto;
        }
        
        .chat-history-item {
            padding: 12px 16px;
            margin: 4px 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            border: 1px solid transparent;
        }
        
        .chat-history-item:hover {
            background-color: #2d2d3d;
        }
        
        .chat-history-item.active {
            background-color: #3d3d5c;
            border-color: #4a90e2;
        }
        
        .chat-history-item .chat-title {
            font-size: 14px;
            font-weight: 500;
            color: #e0e0e0;
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .chat-history-item .chat-time {
            font-size: 11px;
            color: #8e8ea0;
        }
        
        .panel-header {
            padding: 16px;
            border-bottom: 1px solid #3a3a4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-header h3 {
            margin: 0;
            font-size: 16px;
            color: #e0e0e0;
        }
        
        .new-chat-btn {
            width: 100%;
            margin: 12px 0;
            padding: 10px 16px;
        }
        
        .toggle-panel-btn {
            position: fixed;
            left: 480px;
            top: 140px;
            z-index: 1000;
            background-color: #2d2d3d;
            border: 1px solid #3a3a4a;
            border-radius: 0 8px 8px 0;
            padding: 8px 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .toggle-panel-btn:hover {
            background-color: #3d3d5c;
        }
        
        .empty-history {
            text-align: center;
            padding: 40px 20px;
            color: #6c757d;
        }
        
        .delete-chat-btn {
            opacity: 0;
            transition: opacity 0.2s;
            padding: 2px 6px;
            font-size: 12px;
        }
        
        .chat-history-item:hover .delete-chat-btn {
            opacity: 1;
        }
        </style>
        """, unsafe_allow_html=True)


class AISmartAssistantModule:
    # Intent Keywords
    # Priority 1: Improvement intent (generates JSON)
    IMPROVEMENT_KEYWORDS = [
        'improve', 'improvement', 'suggest', 'suggestion', 'recommend', 
        'recommendation', 'enhance', 'enhancement', 'optimize', 'optimization',
        'update', 'modify', 'change', 'better', 'fix', 'upgrade', 'revise',
        'refine', 'boost', 'strengthen', 'develop', 'advance', 'perfect',
        'amend', 'adjust', 'correct', 'tweak', 'overhaul', 'transform',
        # Extraction/matching intent (with file upload)
        'extract', 'match', 'find', 'identify', 'locate', 'search',
        'pull', 'get factors', 'get values', 'read'
    ]
    
    # Priority 2: Analysis intent (no JSON)
    ANALYSIS_KEYWORDS = [
        'analyze', 'analysis', 'check', 'review', 'evaluate', 'evaluation',
        'assess', 'assessment', 'examine', 'audit', 'inspect', 'overview'
    ]
    
    # Priority 3: Question/Explain intent (no JSON)
    QUESTION_KEYWORDS = [
        'what', 'explain', 'how', 'why', 'describe', 'tell me', 'compare',
        'difference', 'meaning', 'define', 'clarify'
    ]
    
    # Topic keywords
    FORMULA_KEYWORDS = [
        'formula', 'formulas', 'calculation', 'calculations', 'method', 
        'methodology', 'accuracy', 'equation', 'compute', 'computing'
    ]
    
    GAP_KEYWORDS = [
        'gap', 'gaps', 'missing', 'source', 'sources', 'coverage', 
        'incomplete', 'lacking', 'absent', 'inventory', 'scope'
    ]
    
    FACTOR_KEYWORDS = [
        'factor', 'factors', 'emission factor', 'gwp', 'coefficient',
        'extract', 'match', 'matching'
    ]
    
    STANDARD_KEYWORDS = [
        'iso', 'ghg protocol', 'greenhouse gas protocol', 'ipcc',
        'compliance', 'compliant', 'standard', 'standards',
        '14064', '14067', '14083', 'regulation', 'requirement', 'requirements',
        'scope 1', 'scope 2', 'scope 3', 'corporate standard', 'reporting'
    ]
    
    # Download confirmation keywords
    DOWNLOAD_CONFIRM_KEYWORDS = [
        # English - affirmative
        'yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay', 'alright', 'right',
        'correct', 'indeed', 'absolutely', 'definitely', 'certainly', 'of course',
        'please', 'go ahead', 'proceed', 'continue', 'do it', 'make it',
        # English - download specific
        'download', 'get', 'give', 'send', 'export', 'save', 'generate',
        'create', 'produce', 'provide', 'show', 'want', 'need', 'like',
        # English - file specific
        'file', 'json', 'the file', 'that file', 'the json', 'that json',
        # Phrases
        'i want', 'i need', 'i would like', "i'd like", 'give me', 'send me',
        'let me', 'can i', 'may i', 'please download', 'yes please',
        'that would be great', 'that works', 'sounds good', 'perfect',
        'great', 'good', 'nice', 'cool', 'awesome', 'excellent', 'wonderful',
        # Short confirmations
        'y', 'ye', 'k', 'kk', 'si', 'ja', 'oui', 'hai'
    ]
    
    # Download decline keywords
    DOWNLOAD_DECLINE_KEYWORDS = [
        'no', 'nope', 'nah', 'not', 'don\'t', 'dont', 'do not', 
        'never', 'cancel', 'skip', 'ignore', 'forget', 'stop',
        'not now', 'later', 'maybe later', 'not yet', 'hold',
        'wait', 'pause', 'decline', 'reject', 'refuse', 'pass',
        'no thanks', 'no thank you', 'not interested', 'unnecessary',
        'not needed', 'don\'t need', 'dont need', 'i\'m good', 'im good',
        'that\'s okay', 'thats okay', 'nevermind', 'never mind'
    ]
    
    def __init__(self, manager):
        self.manager = manager
        self.interface = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_api_key" not in st.session_state:
            st.session_state.chat_api_key = None
        if "extracted_updates" not in st.session_state:
            st.session_state.extracted_updates = {}
        if "analysis_json" not in st.session_state:
            st.session_state.analysis_json = {}
        if "pending_file" not in st.session_state:
            st.session_state.pending_file = None
        # Key for file uploader to force reset after use
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = 0
        # New session states for download confirmation flow
        if "pending_json_generation" not in st.session_state:
            st.session_state.pending_json_generation = {}
        if "awaiting_download_confirmation" not in st.session_state:
            st.session_state.awaiting_download_confirmation = False
        # Chat history management
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  # List of saved conversations
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None
        if "chat_panel_open" not in st.session_state:
            st.session_state.chat_panel_open = False
    
    def display(self):
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not st.session_state.chat_api_key:
            self.display_api_setup()
        else:
            self.display_chat_interface()
    
    def display_api_setup(self):
        if hasattr(st.session_state, 'api_key') and st.session_state.api_key:
            st.session_state.chat_api_key = st.session_state.api_key
            st.rerun()
        
        with st.form("api_key_form"):
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key to enable AI features"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button("Connect", use_container_width=True, type="primary")
            
            if submitted and api_key:
                st.session_state.chat_api_key = api_key
                st.session_state.api_key = api_key
                st.rerun()
    
    def display_chat_interface(self):
        if not self.interface:
            self.interface = ChatInterface(
                api_key=st.session_state.chat_api_key,
                manager=self.manager
            )
        
        self.interface.add_chat_specific_styles()
        
        # Check if we have any chat history or current messages (show toolbar only then)
        has_conversations = st.session_state.chat_messages or st.session_state.chat_history
        
        if has_conversations:
            # Top toolbar with history toggle and new chat button
            toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([2, 5, 2])
            
            with toolbar_col1:
                # History panel toggle button with icon
                if st.session_state.chat_panel_open:
                    btn_label = "Recent Chats <"
                else:
                    btn_label = "Recent Chats >"
                
                if st.button(btn_label, key="toggle_history_panel", use_container_width=True):
                    st.session_state.chat_panel_open = not st.session_state.chat_panel_open
                    st.rerun()
            
            with toolbar_col3:
                # New chat button
                if st.button("+ New Chat", key="new_chat_top", use_container_width=True):
                    self.start_new_chat()
        
        # Main layout
        if has_conversations and st.session_state.chat_panel_open:
            # Two column layout when panel is open
            history_col, main_col = st.columns([1, 3])
            
            with history_col:
                self.display_history_panel()
            
            with main_col:
                self._display_main_chat_area()
        else:
            # Full width when panel is closed or no conversations yet
            self._display_main_chat_area()
    
    def _display_main_chat_area(self):
        if not st.session_state.chat_messages:
            self.display_welcome_screen()
        else:
            # Create a scrollable container for chat messages with fixed height
            chat_container = st.container(height=500)
            with chat_container:
                self.display_chat_history()
        
        # Show extracted updates if any (outside scrollable area)
        if st.session_state.extracted_updates:
            self.display_extracted_updates()
        
        # Input area (fixed at bottom, outside scrollable area)
        self.display_chat_input()
        
        # Bottom button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                self.clear_current_chat()
    
    # Chat History Management Methods
    
    def display_history_panel(self):
        st.markdown("**Saved Chats**")
        
        # Create scrollable container for chat history list
        history_container = st.container(height=450)
        
        with history_container:
            # Display saved chats
            if st.session_state.chat_history:
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    chat_idx = len(st.session_state.chat_history) - 1 - i
                    is_active = chat.get("id") == st.session_state.current_chat_id
                    
                    # Chat item container
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        # Chat item button
                        btn_type = "primary" if is_active else "secondary"
                        title = chat.get("title", "Untitled")
                        display_title = title[:25] + "..." if len(title) > 25 else title
                        if st.button(
                            display_title,
                            key=f"chat_item_{chat_idx}",
                            use_container_width=True,
                            type=btn_type
                        ):
                            self.load_chat(chat_idx)
                    
                    with col2:
                        # Delete button
                        if st.button("🗑", key=f"delete_chat_{chat_idx}"):
                            self.delete_chat(chat_idx)
                    
                    # Show timestamp below
                    if chat.get("updated_at"):
                        st.caption(f"  {chat['updated_at']}")
            else:
                st.caption("No saved chats yet. Start a conversation and it will be saved automatically.")
    
    def generate_chat_title(self, messages: List[Dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Take first 40 characters
                title = content[:40].strip()
                if len(content) > 40:
                    title += "..."
                return title if title else "New Chat"
        return "New Chat"
    
    def save_current_chat(self):
        if not st.session_state.chat_messages:
            return
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = self.generate_chat_title(st.session_state.chat_messages)
        
        chat_data = {
            "id": st.session_state.current_chat_id or str(uuid.uuid4()),
            "title": title,
            "messages": st.session_state.chat_messages.copy(),
            "updated_at": current_time,
            "extracted_updates": st.session_state.extracted_updates.copy() if st.session_state.extracted_updates else {}
        }
        
        # Check if updating existing chat or creating new one
        if st.session_state.current_chat_id:
            # Update existing chat
            for i, chat in enumerate(st.session_state.chat_history):
                if chat.get("id") == st.session_state.current_chat_id:
                    st.session_state.chat_history[i] = chat_data
                    return
        
        # Add new chat
        st.session_state.current_chat_id = chat_data["id"]
        st.session_state.chat_history.append(chat_data)
    
    def start_new_chat(self):
        # Save current chat if it has messages
        if st.session_state.chat_messages:
            self.save_current_chat()
        
        # Reset to new chat
        st.session_state.chat_messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.extracted_updates = {}
        st.session_state.analysis_json = {}
        st.session_state.pending_file = None
        st.session_state.pending_json_generation = {}
        st.session_state.awaiting_download_confirmation = False
        st.session_state.file_uploader_key += 1
        st.rerun()
    
    def load_chat(self, chat_index: int):
        if 0 <= chat_index < len(st.session_state.chat_history):
            # Save current chat first if it has messages
            if st.session_state.chat_messages and st.session_state.current_chat_id:
                self.save_current_chat()
            
            # Load selected chat
            chat = st.session_state.chat_history[chat_index]
            st.session_state.chat_messages = chat.get("messages", []).copy()
            st.session_state.current_chat_id = chat.get("id")
            st.session_state.extracted_updates = chat.get("extracted_updates", {}).copy()
            st.session_state.analysis_json = {}
            st.session_state.pending_file = None
            st.session_state.file_uploader_key += 1
            st.rerun()
    
    def delete_chat(self, chat_index: int):
        if 0 <= chat_index < len(st.session_state.chat_history):
            deleted_chat = st.session_state.chat_history[chat_index]
            st.session_state.chat_history.pop(chat_index)
            
            # If deleting current chat, clear the interface
            if deleted_chat.get("id") == st.session_state.current_chat_id:
                st.session_state.chat_messages = []
                st.session_state.current_chat_id = None
                st.session_state.extracted_updates = {}
            
            st.rerun()
    
    def clear_current_chat(self):
        st.session_state.chat_messages = []
        st.session_state.extracted_updates = {}
        st.session_state.analysis_json = {}
        st.session_state.pending_file = None
        st.session_state.pending_json_generation = {}
        st.session_state.awaiting_download_confirmation = False
        st.session_state.file_uploader_key += 1
        st.session_state.current_chat_id = None
        st.rerun()

    def display_welcome_screen(self):
        st.markdown("""
        <div class="welcome-container">
            <h2>AI Smart Assistant</h2>
            <p style="color: #8e8ea0; margin-bottom: 2rem;">
                Carbon emission model analysis and ISO 14064-1 compliance support
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Analysis only - no JSON
            if st.button("Model Analysis", use_container_width=True):
                self.handle_query("Model Analysis")
            
            # Improvement - generates JSON (now with confirmation flow)
            if st.button("Formula Improvement", use_container_width=True):
                self.handle_query("Formula Improvement")
        
        with col2:
            # Analysis only - no JSON
            if st.button("Standards Compliance", use_container_width=True):
                self.handle_query("Standards Compliance Check")
            
            # Improvement - generates JSON (now with confirmation flow)
            if st.button("Gap Analysis", use_container_width=True):
                self.handle_query("Gap Analysis")
    
    def display_chat_history(self):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                # Check if message contains download links
                if msg.get("has_download_link"):
                    # Handle multiple download links
                    if msg.get("download_links"):
                        # Display content before links
                        content = msg["content"]
                        st.markdown(content)
                        
                        # Display each download link
                        for link_info in msg["download_links"]:
                            self.display_inline_download_link(
                                link_info["json_data"],
                                link_info["json_type"],
                                link_info["filename"]
                            )
                            # Display instruction for this specific file
                            st.markdown(f"\n{link_info['instruction']}\n")
                    
                    # Handle single download link (backward compatibility)
                    elif msg.get("json_data"):
                        content = msg["content"]
                        # Remove placeholder if exists
                        content = content.replace("[DOWNLOAD_LINK_PLACEHOLDER]", "")
                        st.markdown(content)
                        
                        self.display_inline_download_link(
                            msg["json_data"],
                            msg.get("json_type", "data"),
                            msg.get("filename", "data.json")
                        )
                else:
                    st.markdown(msg["content"])
                
                if msg.get("file"):
                    st.markdown(f'<div class="file-attachment">Attached: {msg["file"]}</div>', 
                              unsafe_allow_html=True)
    
    def display_inline_download_link(self, json_data: Dict, json_type: str, filename: str):
        json_str = json.dumps(json_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        
        # Create download link HTML
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="download-link">Download {filename}</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def display_extracted_updates(self):
        updates_data = st.session_state.extracted_updates
        
        if not updates_data:
            return
        
        st.markdown("---")
        st.markdown("### Extracted Updates")
        
        if "matched" in updates_data and updates_data["matched"]:
            st.markdown("**Matched Variables**")
            for match in updates_data["matched"]:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.text(f"Variable: {match['stub']}")
                with col2:
                    st.text(f"Current: {match.get('current_value', 'N/A')}")
                with col3:
                    st.text(f"New: {match['new_value']}")
                with col4:
                    if st.button("Apply", key=f"apply_{match['stub']}"):
                        self.apply_single_update(match)
        
        if "new" in updates_data and updates_data["new"]:
            st.markdown("**New Parameters**")
            for new_param in updates_data["new"]:
                st.text(f"{new_param['name']}: {new_param['value']}")
        
        if "matched" in updates_data and updates_data["matched"]:
            if st.button("Apply All Matched Updates", use_container_width=True, type="primary"):
                self.apply_all_updates()
    
    def display_chat_input(self):
        st.markdown("---")
        
        # Use dynamic key to reset file uploader after file is used
        uploaded_file = st.file_uploader(
            "Upload Document (Optional)",
            type=['pdf', 'txt', 'json', 'csv'],
            help="Upload documents for analysis",
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        
        if uploaded_file is not None:
            st.session_state.pending_file = uploaded_file
            st.info(f"📎 File ready: {uploaded_file.name}")
        
        with st.expander("Current Emission Factors", expanded=False):
            factors = self.interface.get_emission_factors_only() if self.interface else {}
            if factors:
                models_dict = {}
                for stub, info in factors.items():
                    model = info['model']
                    if model not in models_dict:
                        models_dict[model] = []
                    models_dict[model].append({
                        "Variable": stub,
                        "Name": info['name'],
                        "Value": f"{info['value']:.6f}" if isinstance(info['value'], (int, float)) else info['value']
                    })
                
                for model_name, vars_list in models_dict.items():
                    st.markdown(f"**{model_name}**")
                    df = pd.DataFrame(vars_list)
                    st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.text("No emission factors found")
        
        user_input = st.chat_input("Enter your question or request...")
        
        if user_input:
            self.handle_query(user_input)
    
    def handle_query(self, query: str):
        """Handle query from button or input"""
        self.process_user_message(query, st.session_state.pending_file)
    
    # Intent Detection Methods
    def detect_intent(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        # Priority 1: Check for improvement intent (HIGHEST PRIORITY)
        if any(keyword in prompt_lower for keyword in self.IMPROVEMENT_KEYWORDS):
            return "improvement"
        
        # Priority 2: Check for analysis intent
        if any(keyword in prompt_lower for keyword in self.ANALYSIS_KEYWORDS):
            return "analysis"
        
        # Priority 3: Check for question intent
        if any(keyword in prompt_lower for keyword in self.QUESTION_KEYWORDS):
            return "question"
        
        # Priority 4: General chat
        return "general"
    
    def detect_topic(self, prompt: str, has_file: bool = False) -> str:
        prompt_lower = prompt.lower()
        
        # Priority 1: Formula topic
        if any(keyword in prompt_lower for keyword in self.FORMULA_KEYWORDS):
            return "formula"
        
        # Priority 2: Gap/Sources topic
        if any(keyword in prompt_lower for keyword in self.GAP_KEYWORDS):
            return "gap"
        
        # Priority 3: Factor topic (boost priority if file attached)
        if any(keyword in prompt_lower for keyword in self.FACTOR_KEYWORDS):
            return "factor"
        
        # If file is attached and no other topic detected, default to factor
        if has_file and "document" in prompt_lower:
            return "factor"
        
        # Priority 4: Standards/Compliance topic
        if any(keyword in prompt_lower for keyword in self.STANDARD_KEYWORDS):
            return "standard"
        
        # Default: General/Model
        return "model"
    
    def is_download_confirmation(self, message: str) -> bool:
        message_lower = message.lower().strip()
        
        # First check for explicit decline
        for keyword in self.DOWNLOAD_DECLINE_KEYWORDS:
            if keyword in message_lower:
                return False
        
        # Then check for confirmation
        for keyword in self.DOWNLOAD_CONFIRM_KEYWORDS:
            if keyword in message_lower:
                return True
        
        # If message is very short (1-3 words), be more lenient
        word_count = len(message_lower.split())
        if word_count <= 3:
            # Single word or short phrase - likely a response to the question
            # Check if it's not a clear decline
            if not any(neg in message_lower for neg in ['no', 'not', 'don', 'nope', 'nah']):
                # Could be affirmative
                return True
        
        return False
    
    def is_download_decline(self, message: str) -> bool:
        message_lower = message.lower().strip()
        
        for keyword in self.DOWNLOAD_DECLINE_KEYWORDS:
            if keyword in message_lower:
                return True
        
        return False
    
    def process_user_message(self, message: str, uploaded_file):
        # Track if file was uploaded for later cleanup
        has_uploaded_file = uploaded_file is not None
        
        # Ensure we have a chat_id for this conversation
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())
        
        # Check if we're awaiting download confirmation
        if st.session_state.awaiting_download_confirmation:
            if self.is_download_confirmation(message):
                # User confirmed download
                self.generate_and_provide_json(message)
                return
            elif self.is_download_decline(message):
                # User declined download
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": message,
                    "file": None
                })
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "Understood. The JSON file will not be generated. Feel free to ask if you need anything else!"
                })
                # Reset state
                st.session_state.awaiting_download_confirmation = False
                st.session_state.pending_json_generation = {}
                st.rerun()
                return
            else:
                # User said something else - reset waiting state and continue normally
                st.session_state.awaiting_download_confirmation = False
                st.session_state.pending_json_generation = {}
        
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": message,
            "file": uploaded_file.name if uploaded_file else None
        })
        
        # Build full message with file context
        full_message = message
        has_file = uploaded_file is not None
        
        if has_file:
            file_content = self.interface.build_file_context(uploaded_file)
            full_message += file_content
        
        # Detect intent and topic
        intent = self.detect_intent(message)
        topic = self.detect_topic(message, has_file)
        
        # Clear previous results
        st.session_state.analysis_json = {}
        st.session_state.extracted_updates = {}
        
        # Get conversation history
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.chat_messages[:-1]
        ]
        
        # Generate response based on intent and topic
        with st.spinner("Processing..."):
            response = self.execute_action(intent, topic, full_message, has_file, conversation_history)
        
        if response:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "An error occurred while processing your request."
            })
        
        # Auto-save chat after each message exchange
        self.save_current_chat()
        
        # Reset file uploader if a file was uploaded (increment key to clear the uploader)
        if has_uploaded_file:
            st.session_state.pending_file = None
            st.session_state.file_uploader_key += 1
        
        st.rerun()
    
    def execute_action(self, intent: str, topic: str, full_message: str, 
                       has_file: bool, conversation_history: List[Dict]) -> Optional[str]:
        # IMPROVEMENT INTENT (Ask for JSON download)
        if intent == "improvement":
            
            if topic == "formula":
                if has_file:
                    response = self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_formula_suggestion_prompt()
                    )
                else:
                    response = self.interface.analyze_formula_improvements()
                
                # Store pending JSON generation info
                st.session_state.pending_json_generation = {
                    'type': 'formula',
                    'topic': topic,
                    'response': response,
                    'has_file': has_file,
                    'full_message': full_message
                }
                st.session_state.awaiting_download_confirmation = True
                
                # Append download question to response
                response += self.get_download_question('formula')
                return response
            
            elif topic == "gap":
                if has_file:
                    response = self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_gap_analysis_prompt()
                    )
                else:
                    response = self.interface.analyze_emission_gaps()
                
                # Store pending JSON generation info
                st.session_state.pending_json_generation = {
                    'type': 'source',
                    'topic': topic,
                    'response': response,
                    'has_file': has_file,
                    'full_message': full_message
                }
                st.session_state.awaiting_download_confirmation = True
                
                # Append download question to response
                response += self.get_download_question('source')
                return response
            
            elif topic == "factor":
                if has_file:
                    response = self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_parameter_matching_prompt()
                    )
                    
                    # Check if any factors were found
                    no_factors_found = "NO EMISSION FACTORS FOUND" in response.upper() or \
                                       "no emission factors" in response.lower() or \
                                       "does not contain" in response.lower()
                    
                    if no_factors_found:
                        # Don't ask for download if no factors found
                        return response
                    
                    # Parse for apply buttons
                    extracted = self.interface.parse_extraction_response(response)
                    if extracted:
                        st.session_state.extracted_updates = extracted
                        
                        # Only ask for download if we found matches
                        st.session_state.pending_json_generation = {
                            'type': 'factor',
                            'topic': topic,
                            'response': response,
                            'has_file': has_file,
                            'full_message': full_message
                        }
                        st.session_state.awaiting_download_confirmation = True
                        
                        # Append download question to response
                        response += self.get_download_question('factor')
                    
                    return response
                else:
                    return "Please upload a document to extract and match emission factors."
            
            elif topic == "model":
                # Comprehensive improvement - both formula and gap
                responses = []
                
                formula_response = self.interface.analyze_formula_improvements()
                responses.append(f"### Formula Improvements\n{formula_response}")
                
                gap_response = self.interface.analyze_emission_gaps()
                responses.append(f"### Gap Analysis\n{gap_response}")
                
                combined_response = "\n\n---\n\n".join(responses)
                
                # Store pending JSON generation info for both
                st.session_state.pending_json_generation = {
                    'type': 'model',
                    'topic': topic,
                    'formula_response': formula_response,
                    'gap_response': gap_response,
                    'has_file': has_file,
                    'full_message': full_message
                }
                st.session_state.awaiting_download_confirmation = True
                
                # Append download question
                combined_response += self.get_download_question('model')
                return combined_response
            
            else:
                # Default improvement
                response = self.interface.analyze_formula_improvements()
                
                st.session_state.pending_json_generation = {
                    'type': 'formula',
                    'topic': 'formula',
                    'response': response,
                    'has_file': has_file,
                    'full_message': full_message
                }
                st.session_state.awaiting_download_confirmation = True
                
                response += self.get_download_question('formula')
                return response
        
        # ANALYSIS INTENT (No JSON)
        elif intent == "analysis":
            
            if topic == "formula":
                if has_file:
                    return self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_formula_suggestion_prompt()
                    )
                else:
                    return self.interface.analyze_formula_improvements()
            
            elif topic == "gap":
                if has_file:
                    return self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_gap_analysis_prompt()
                    )
                else:
                    return self.interface.analyze_emission_gaps()
            
            elif topic == "factor":
                if has_file:
                    return self.interface.send_to_gpt(
                        [{"role": "user", "content": full_message}],
                        system_prompt=self.interface.get_parameter_matching_prompt()
                    )
                else:
                    return "Please upload a document to analyze emission factors."
            
            elif topic == "standard":
                return self.interface.check_standards_compliance()
            
            elif topic == "model":
                return self.interface.analyze_model_structure()
            
            else:
                return self.interface.analyze_model_structure()
        
        # QUESTION INTENT (No JSON) 
        elif intent == "question":
            return self.interface.chat(full_message, conversation_history)
        
        # GENERAL CHAT (No JSON) 
        else:
            return self.interface.chat(full_message, conversation_history)
    
    def get_download_question(self, json_type: str) -> str:
        questions = {
            'formula': "\n\n---\n\n**Would you like to download the JSON file for these formula improvements?**\n\nReply **'yes'** or **'download'** to get the file.",
            'source': "\n\n---\n\n**Would you like to download the JSON file for these new emission sources?**\n\nReply **'yes'** or **'download'** to get the file.",
            'factor': "\n\n---\n\n**Would you like to download the JSON file for these emission factor updates?**\n\nReply **'yes'** or **'download'** to get the file.",
            'model': "\n\n---\n\n**Would you like to download the JSON files for these improvements?**\n\nReply **'yes'** or **'download'** to get the files (formula improvements and new emission sources will be provided as separate files)."
        }
        return questions.get(json_type, "\n\n---\n\n**Would you like to download the JSON file?**\n\nReply **'yes'** or **'download'** to get the file.")
    
    def get_upload_instruction(self, json_type: str) -> str:
        instructions = {
            'formula': """**How to use this file:**
1. Go to the sidebar and click **"Model Structure Editor"**
2. Click the **"Edit Formulas"** button
3. Select **"JSON File Upload"** mode
4. Upload this downloaded JSON file
5. Review and confirm the formula changes""",

            'source': """**How to use this file:**
1. Go to the sidebar and click **"Model Structure Editor"**
2. Click the **"Source Management"** button
3. Use the **"Add from JSON"** option
4. Upload this downloaded JSON file
5. Review and confirm the new emission sources""",

            'factor': """**How to use this file:**
1. Go to the sidebar and click **"Quick Data Update"**
2. Click the **"Parameter Management"** button
3. Look for the **"JSON File Upload"** section
4. Upload this downloaded JSON file
5. Review and confirm the parameter updates"""
        }
        return instructions.get(json_type, "Upload this file to the appropriate section in the tool.")
    
    def generate_and_provide_json(self, user_message: str):
        pending = st.session_state.pending_json_generation
        
        if not pending:
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_message,
                "file": None
            })
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Sorry, there was an error generating the JSON file. Please try the analysis again."
            })
            st.session_state.awaiting_download_confirmation = False
            st.rerun()
            return
        
        # Add user confirmation message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message,
            "file": None
        })
        
        json_type = pending.get('type')
        
        if json_type == 'model':
            # Handle model type - generate both formula and source JSONs
            formula_response = pending.get('formula_response', '')
            gap_response = pending.get('gap_response', '')
            
            formula_json = self.interface.generate_formula_improvement_json(formula_response)
            source_json = self.interface.generate_emission_source_json(gap_response)
            
            download_links = []
            response_content = "**JSON files generated successfully!**\n\n"
            
            if formula_json:
                download_links.append({
                    "json_data": formula_json,
                    "json_type": "formula",
                    "filename": "formula_improvements.json",
                    "instruction": self.get_upload_instruction('formula')
                })
                response_content += "**1. Formula Improvements File:**\n\n"
            
            if source_json:
                download_links.append({
                    "json_data": source_json,
                    "json_type": "source",
                    "filename": "new_emission_sources.json",
                    "instruction": self.get_upload_instruction('source')
                })
                if formula_json:
                    response_content += "\n---\n\n**2. New Emission Sources File:**\n\n"
                else:
                    response_content += "**New Emission Sources File:**\n\n"
            
            if download_links:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "has_download_link": True,
                    "download_links": download_links
                })
            else:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "Could not generate JSON files. The analysis may not have produced structured improvements in the expected format. Please review the analysis above for manual implementation."
                })
            
            # Reset state
            st.session_state.awaiting_download_confirmation = False
            st.session_state.pending_json_generation = {}
            st.rerun()
            return
        
        # Handle single JSON types
        if json_type == 'formula':
            response_text = pending.get('response', '')
            json_data = self.interface.generate_formula_improvement_json(response_text)
            filename = "formula_improvements.json"
            
        elif json_type == 'source':
            response_text = pending.get('response', '')
            json_data = self.interface.generate_emission_source_json(response_text)
            filename = "new_emission_sources.json"
            
        elif json_type == 'factor':
            response_text = pending.get('response', '')
            json_data = self.interface.generate_emission_factor_json(response_text)
            filename = "emission_factor_updates.json"
            
        else:
            json_data = None
            filename = "data.json"
        
        if json_data:
            # Create response with download link and instructions
            response_content = f"**JSON file generated successfully!**\n\n"
            
            download_links = [{
                "json_data": json_data,
                "json_type": json_type,
                "filename": filename,
                "instruction": self.get_upload_instruction(json_type)
            }]
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_content,
                "has_download_link": True,
                "download_links": download_links
            })
        else:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Could not generate JSON file. The analysis may not have produced structured improvements in the expected format. Please review the analysis above for manual implementation."
            })
        
        # Reset state
        st.session_state.awaiting_download_confirmation = False
        st.session_state.pending_json_generation = {}
        st.rerun()
    
    def apply_single_update(self, update: Dict):
        try:
            variable_stub = update['stub']
            new_value = float(update['new_value'])
            
            for model in self.manager.models.values():
                if variable_stub in model.data_modes:
                    old_value = model.data_modes[variable_stub].value
                    model.data_modes[variable_stub].value = new_value
                    st.success(f"Updated {variable_stub}: {old_value} -> {new_value}")
                    
                    self.manager.calculate_all()
                    
                    if "matched" in st.session_state.extracted_updates:
                        st.session_state.extracted_updates["matched"] = [
                            m for m in st.session_state.extracted_updates["matched"] 
                            if m['stub'] != variable_stub
                        ]
                    
                    st.rerun()
                    return
            
            st.error(f"Variable {variable_stub} not found")
        except Exception as e:
            st.error(f"Error applying update: {str(e)}")
    
    def apply_all_updates(self):
        updates_data = st.session_state.extracted_updates
        
        if "matched" not in updates_data:
            return
        
        applied = []
        failed = []
        
        for update in updates_data["matched"]:
            try:
                variable_stub = update['stub']
                new_value = float(update['new_value'])
                
                updated = False
                for model in self.manager.models.values():
                    if variable_stub in model.data_modes:
                        old_value = model.data_modes[variable_stub].value
                        model.data_modes[variable_stub].value = new_value
                        applied.append(f"{variable_stub}: {old_value} -> {new_value}")
                        updated = True
                        break
                
                if not updated:
                    failed.append(f"{variable_stub} not found")
                    
            except Exception as e:
                failed.append(f"{variable_stub}: {str(e)}")
        
        if applied:
            self.manager.calculate_all()
            st.success(f"Applied {len(applied)} updates")
            for update in applied:
                st.write(f"- {update}")
        
        if failed:
            st.warning(f"{len(failed)} updates failed")
            for failure in failed:
                st.write(f"- {failure}")
        
        st.session_state.extracted_updates = {}
        st.rerun()