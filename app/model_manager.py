import json
import os
import ast
from copy import deepcopy
import random

class DataMode:
    def __init__(self, name: str, stub: str, value: float):
        self.name = name
        self.stub = stub
        self.value = value

    def __repr__(self):
        return f"DataMode(Name={self.name}, Stub={self.stub}, Value={self.value})"

class FormulaMode:
    def __init__(self, name: str, stub: str, formula: str):
        self.name = name
        self.stub = stub
        self.formula = formula
        self.result = None

    def evaluate(self, variables: dict):
        try:
            self.result = eval(self.formula, {}, variables)
        except Exception as e:
            print(f"Error evaluating {self.name}: {e}")
        return self.result

    def __repr__(self):
        return f"FormulaMode(Name={self.name}, Stub={self.stub}, Formula={self.formula}, Result={self.result})"

def extract_variables(expr: str) -> set:
    return {node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)}


def get_model_scope(model_name: str) -> str:
    if model_name.startswith("Scope1"):
        return "Scope1"
    elif model_name.startswith("Scope2"):
        return "Scope2"
    elif model_name.startswith("Scope3"):
        return "Scope3"
    else:
        return "Other"


def is_total_model(model_name: str) -> bool:
    return model_name.endswith("_Total") or model_name == "GrandTotal"


def get_scope_models(models: dict, scope: str, include_totals: bool = True) -> list:
    result = []
    for model_name in models:
        if get_model_scope(model_name) == scope:
            if include_totals or not is_total_model(model_name):
                result.append(model_name)
    return result


class EmissionModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_modes = {}
        self.formula_modes = {}

    def add_data_mode(self, name, stub, value):
        mode = DataMode(name, stub, value)
        self.data_modes[stub] = mode

    def add_formula_mode(self, name, stub, formula):
        mode = FormulaMode(name, stub, formula)
        self.formula_modes[stub] = mode

    def calculate(self, external_vars=None):
        variables = {stub: mode.value for stub, mode in self.data_modes.items()}
        if external_vars:
            variables.update(external_vars)
        evaluated = set()
        for _ in range(len(self.formula_modes)):
            updated = False
            for stub, mode in self.formula_modes.items():
                if stub in evaluated:
                    continue
                result = mode.evaluate(variables)
                if result is not None:
                    variables[stub] = result
                    evaluated.add(stub)
                    updated = True
            if not updated:
                break
        return variables

    def __repr__(self):
        lines = [f"Model: {self.model_name}"]
        for mode in self.formula_modes.values():
            if mode.result is not None:
                lines.append(f"  {mode.stub:25s} = {mode.result:.8f}")
            else:
                lines.append(f"  {mode.stub:25s} = None")
        return "\n".join(lines)

class ModelManager:
    def __init__(self, formula_file=None):
        self.models = {}
        self.formula_file = formula_file
        self.model_loaded = False  # Track if a model is loaded
        
        # Do NOT auto-load anything - start empty

    def load_formulas(self):
        try:
            with open(self.formula_file, "r", encoding="utf-8") as f:
                formulas = json.load(f)
                for model_name, formula_dict in formulas.items():
                    model = EmissionModel(model_name)
                    for formula_name, data in formula_dict.items():
                        model.add_formula_mode(formula_name, data["stub"], data["formula"])
                    self.models[model_name] = model
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading formulas: {e}")

    def load_complete_model(self, file_content):
        try:
            # Handle both dict and file path
            if isinstance(file_content, dict):
                model_data = file_content
            elif isinstance(file_content, str):
                with open(file_content, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
            else:
                # Assume it's a file-like object (uploaded file)
                model_data = json.load(file_content)
            
            # Clear existing models
            self.models = {}
            
            # Statistics
            stats = {
                "models": 0,
                "formulas": 0,
                "parameters": 0
            }
            
            # Parse each model
            for model_name, model_content in model_data.items():
                # Skip metadata fields (starting with _)
                if model_name.startswith("_"):
                    continue
                
                model = EmissionModel(model_name)
                
                # Load formulas
                if "formulas" in model_content:
                    for formula_name, formula_data in model_content["formulas"].items():
                        if "stub" in formula_data and "formula" in formula_data:
                            model.add_formula_mode(
                                formula_name, 
                                formula_data["stub"], 
                                formula_data["formula"]
                            )
                            stats["formulas"] += 1
                
                # Load data
                if "data" in model_content:
                    for stub, data_info in model_content["data"].items():
                        if "name" in data_info and "value" in data_info:
                            model.add_data_mode(
                                data_info["name"],
                                stub,
                                data_info["value"]
                            )
                            stats["parameters"] += 1
                
                self.models[model_name] = model
                stats["models"] += 1
            
            # Calculate all models
            self.calculate_all()
            self.model_loaded = True
            
            return True, "Model loaded successfully", stats
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", {}
        except Exception as e:
            return False, f"Error loading model: {str(e)}", {}

    def clear_models(self):
        self.models = {}
        self.model_loaded = False
        return True, "All models cleared"

    def export_complete_model(self):
        export_data = {}
        
        for model_name, model in self.models.items():
            model_export = {
                "formulas": {},
                "data": {}
            }
            
            # Export formulas
            for stub, formula_mode in model.formula_modes.items():
                model_export["formulas"][formula_mode.name] = {
                    "stub": stub,
                    "formula": formula_mode.formula
                }
            
            # Export data
            for stub, data_mode in model.data_modes.items():
                model_export["data"][stub] = {
                    "name": data_mode.name,
                    "value": data_mode.value
                }
            
            export_data[model_name] = model_export
        
        return export_data

    def is_model_loaded(self):
        return self.model_loaded and len(self.models) > 0
    
    def get_models_by_scope(self, scope: str, include_totals: bool = True) -> dict:
        result = {}
        for model_name, model in self.models.items():
            if get_model_scope(model_name) == scope:
                if include_totals or not is_total_model(model_name):
                    result[model_name] = model
        return result
    
    def get_scope_structure(self) -> dict:
        structure = {
            "Scope1": {"models": [], "total_model": None},
            "Scope2": {"models": [], "total_model": None},
            "Scope3": {"models": [], "total_model": None},
            "Other": {"models": [], "total_model": None}
        }
        
        for model_name in self.models:
            scope = get_model_scope(model_name)
            
            if is_total_model(model_name):
                if model_name == "GrandTotal":
                    structure["Other"]["total_model"] = model_name
                else:
                    structure[scope]["total_model"] = model_name
            else:
                structure[scope]["models"].append(model_name)
        
        return structure

    def get_model(self, model_name):
        return self.models.get(model_name, None)

    def calculate_all(self):
        global_vars = {}
        for model in self.models.values():
            result = model.calculate(global_vars)
            global_vars.update(result)

    def summary_report(self, target_stubs=None):
        lines = ["Final Summary Report:"]
        for model in self.models.values():
            for stub, mode in model.formula_modes.items():
                if (target_stubs is None or stub in target_stubs) and mode.result is not None:
                    lines.append(f"{stub:25s} = {mode.result:.6f}")
        return "\n".join(lines)

    def visualize_model(self, output_path="data/graphs/model_structure"):
        try:
            from graphviz import Digraph
        except ImportError:
            print("Graphviz not installed. Please install with: pip install graphviz")
            return None
        
        try:
            import os
            # Ensure output directory exists
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create directed graph
            dot = Digraph(name="ModelStructure", format="png")
            dot.attr(rankdir="LR")  # Left to right layout
            dot.attr('node', fontname='Arial', fontsize='10')
            dot.attr('edge', fontname='Arial', fontsize='8')
            
            # Collect all variables first
            all_vars = {}
            for model in self.models.values():
                for stub, mode in model.data_modes.items():
                    all_vars[stub] = mode.value
            
            # Calculate all formulas to get results
            for model in self.models.values():
                for stub, mode in model.formula_modes.items():
                    mode.evaluate(all_vars)
                    all_vars[stub] = mode.result
            
            # Create subgraphs for each model
            for model in self.models.values():
                with dot.subgraph(name=f"cluster_{model.model_name}") as sub:
                    sub.attr(label=f"Model: {model.model_name}", 
                            style="dashed", 
                            color="blue",
                            fontsize="12",
                            fontname="Arial Bold")
                    
                    # Add data nodes (input parameters)
                    for stub, mode in model.data_modes.items():
                        label = f"{stub}\\n[input]\\n= {mode.value}"
                        sub.node(stub, 
                                label=label, 
                                shape="box", 
                                style="filled", 
                                fillcolor="#f0f8ff",  # Light blue
                                color="#4169e1")      # Border blue
                    
                    # Add formula nodes (computed values)
                    for stub, mode in model.formula_modes.items():
                        result_text = f"{mode.result:.4f}" if mode.result is not None else "None"
                        label = f"{stub}\\n[computed]\\n= {result_text}"
                        sub.node(stub, 
                                label=label, 
                                shape="ellipse", 
                                style="filled", 
                                fillcolor="#d0f0c0",  # Light green
                                color="#32cd32")      # Border green
                        
                        # Add edges from dependencies to this formula
                        dependencies = extract_variables(mode.formula)
                        for var in dependencies:
                            if var in all_vars:  # Only add edge if variable exists
                                dot.edge(var, stub, color="#666666")
            
            # Render the graph
            dot.render(output_path, cleanup=True)
            print(f"Model structure graph saved to: {output_path}.png")
            return f"{output_path}.png"
            
        except Exception as e:
            print(f"Error generating graph: {str(e)}")
            return None

    def __repr__(self):
        divider = "\n" + "="*50 + "\n"
        return divider.join([str(model) for model in self.models.values()])


# Template for users to download
MODEL_TEMPLATE = {
  "_README": {
    "description": "Carbon Emission Model Template - Customize this template for your organization",
    "version": "2.0",
    "naming_convention": {
      "Scope1_*": "Direct emissions (e.g., Scope1_FixedCombustion, Scope1_Mobile, Scope1_Fugitive)",
      "Scope2_*": "Indirect emissions from purchased energy (e.g., Scope2_Electricity, Scope2_Steam)",
      "Scope3_*": "Other indirect emissions (e.g., Scope3_Water, Scope3_Transport, Scope3_Waste)",
      "*_Total": "Summary models for each scope (e.g., Scope1_Total, Scope2_Total)",
      "GrandTotal": "Organization total emissions"
    },
    "structure": {
      "formulas": "Calculation formulas using stub variables",
      "data": "Input parameters with name and value"
    },
    "supported_operators": [
      "+ (addition)",
      "- (subtraction)", 
      "* (multiplication)",
      "/ (division)",
      "** (power/exponent)",
      "() (parentheses for grouping)"
    ],
    "notes": [
      "Model names MUST follow the naming convention (Scope1_*, Scope2_*, Scope3_*)",
      "The stub in 'data' section must match variables used in 'formulas'",
      "Formulas can reference results from other formulas",
      "Models are calculated in order, so later models can use results from earlier models"
    ]
  },

  "Scope1_SourceA": {
    "_description": "Example: Addition and Subtraction - Fixed combustion emissions",
    "formulas": {
      "Fuel Total Consumption": {
        "stub": "FuelTotal",
        "formula": "Fuel_TypeA + Fuel_TypeB + Fuel_TypeC"
      },
      "Net Emission After Offset": {
        "stub": "NetEmission_A",
        "formula": "FuelTotal - Carbon_Offset"
      },
      "Source A CO2e": {
        "stub": "SourceA_CO2e",
        "formula": "NetEmission_A + Additional_Emission"
      }
    },
    "data": {
      "Fuel_TypeA": {
        "name": "Fuel Type A Consumption (kg)",
        "value": 1000
      },
      "Fuel_TypeB": {
        "name": "Fuel Type B Consumption (kg)",
        "value": 500
      },
      "Fuel_TypeC": {
        "name": "Fuel Type C Consumption (kg)",
        "value": 250
      },
      "Carbon_Offset": {
        "name": "Carbon Offset Credits (kg CO2e)",
        "value": 100
      },
      "Additional_Emission": {
        "name": "Additional Emission (kg CO2e)",
        "value": 50
      }
    }
  },

  "Scope1_SourceB": {
    "_description": "Example: Multiplication - Mobile combustion with emission factors",
    "formulas": {
      "Vehicle Emission": {
        "stub": "VehicleEmission",
        "formula": "Fuel_Consumption * Emission_Factor"
      },
      "Equipment Emission": {
        "stub": "EquipmentEmission", 
        "formula": "Equipment_Hours * Fuel_Rate * Equipment_EF"
      },
      "Source B CO2e": {
        "stub": "SourceB_CO2e",
        "formula": "VehicleEmission + EquipmentEmission"
      }
    },
    "data": {
      "Fuel_Consumption": {
        "name": "Vehicle Fuel Consumption (liters)",
        "value": 5000
      },
      "Emission_Factor": {
        "name": "Fuel Emission Factor (kg CO2e/liter)",
        "value": 2.31
      },
      "Equipment_Hours": {
        "name": "Equipment Operating Hours",
        "value": 200
      },
      "Fuel_Rate": {
        "name": "Fuel Consumption Rate (liters/hour)",
        "value": 15
      },
      "Equipment_EF": {
        "name": "Equipment Emission Factor (kg CO2e/liter)",
        "value": 2.68
      }
    }
  },

  "Scope1_Total": {
    "_description": "Scope 1 Summary - References results from all Scope1 sources",
    "formulas": {
      "Scope 1 Total": {
        "stub": "Scope1_Total",
        "formula": "SourceA_CO2e + SourceB_CO2e"
      }
    },
    "data": {}
  },

  "Scope2_Electricity": {
    "_description": "Example: Division - Electricity with efficiency and loss calculations",
    "formulas": {
      "Gross Electricity": {
        "stub": "GrossElectricity",
        "formula": "Metered_Electricity / Meter_Accuracy"
      },
      "Transmission Loss": {
        "stub": "TransmissionLoss",
        "formula": "GrossElectricity * Loss_Rate / 100"
      },
      "Net Electricity CO2e": {
        "stub": "NetElectricity_CO2e",
        "formula": "(GrossElectricity + TransmissionLoss) * Grid_EF"
      }
    },
    "data": {
      "Metered_Electricity": {
        "name": "Metered Electricity Usage (kWh)",
        "value": 100000
      },
      "Meter_Accuracy": {
        "name": "Meter Accuracy Factor (e.g., 0.98 = 98% accurate)",
        "value": 0.98
      },
      "Loss_Rate": {
        "name": "Transmission Loss Rate (%)",
        "value": 5
      },
      "Grid_EF": {
        "name": "Grid Emission Factor (kg CO2e/kWh)",
        "value": 0.5
      }
    }
  },

  "Scope2_Steam": {
    "_description": "Example: Complex division and multiplication - Purchased steam/heat",
    "formulas": {
      "Steam Energy Content": {
        "stub": "SteamEnergy",
        "formula": "Steam_Mass * Enthalpy_Difference / 1000"
      },
      "Boiler Input Energy": {
        "stub": "BoilerInput",
        "formula": "SteamEnergy / Boiler_Efficiency"
      },
      "Steam CO2e": {
        "stub": "Steam_CO2e",
        "formula": "BoilerInput * Fuel_EF"
      }
    },
    "data": {
      "Steam_Mass": {
        "name": "Purchased Steam Mass (kg)",
        "value": 50000
      },
      "Enthalpy_Difference": {
        "name": "Enthalpy Difference (kJ/kg)",
        "value": 2675
      },
      "Boiler_Efficiency": {
        "name": "Boiler Efficiency (decimal, e.g., 0.85)",
        "value": 0.85
      },
      "Fuel_EF": {
        "name": "Fuel Emission Factor (kg CO2e/GJ)",
        "value": 56.1
      }
    }
  },

  "Scope2_Total": {
    "_description": "Scope 2 Summary",
    "formulas": {
      "Scope 2 Total": {
        "stub": "Scope2_Total",
        "formula": "NetElectricity_CO2e + Steam_CO2e"
      }
    },
    "data": {}
  },

  "Scope3_Transport": {
    "_description": "Example: Power/Exponent - Distance-based calculations with scaling",
    "formulas": {
      "Base Transport Emission": {
        "stub": "BaseTransport",
        "formula": "Distance * Weight * Transport_EF"
      },
      "Scaling Factor": {
        "stub": "ScalingFactor",
        "formula": "Base_Scale ** Scale_Exponent"
      },
      "Scaled Transport CO2e": {
        "stub": "ScaledTransport_CO2e",
        "formula": "BaseTransport * ScalingFactor"
      }
    },
    "data": {
      "Distance": {
        "name": "Transport Distance (km)",
        "value": 500
      },
      "Weight": {
        "name": "Cargo Weight (tonnes)",
        "value": 10
      },
      "Transport_EF": {
        "name": "Transport Emission Factor (kg CO2e/tonne-km)",
        "value": 0.1
      },
      "Base_Scale": {
        "name": "Base Scaling Value",
        "value": 1.05
      },
      "Scale_Exponent": {
        "name": "Scaling Exponent (e.g., 2 for square)",
        "value": 2
      }
    }
  },

  "Scope3_Waste": {
    "_description": "Example: Square root and complex formulas - Waste decomposition model",
    "formulas": {
      "Decomposition Factor": {
        "stub": "DecompFactor",
        "formula": "(Waste_Mass * Organic_Content) ** 0.5"
      },
      "Methane Generation": {
        "stub": "MethaneGen",
        "formula": "DecompFactor * Methane_Potential * (1 - Recovery_Rate / 100)"
      },
      "Waste CO2e": {
        "stub": "Waste_CO2e",
        "formula": "MethaneGen * CH4_GWP"
      }
    },
    "data": {
      "Waste_Mass": {
        "name": "Total Waste Mass (tonnes)",
        "value": 1000
      },
      "Organic_Content": {
        "name": "Organic Content Fraction (decimal)",
        "value": 0.6
      },
      "Methane_Potential": {
        "name": "Methane Generation Potential (kg CH4/tonne)",
        "value": 50
      },
      "Recovery_Rate": {
        "name": "Methane Recovery Rate (%)",
        "value": 30
      },
      "CH4_GWP": {
        "name": "Methane Global Warming Potential",
        "value": 28
      }
    }
  },

  "Scope3_Water": {
    "_description": "Example: Multiple operations combined - Water treatment emissions",
    "formulas": {
      "Treatment Energy": {
        "stub": "TreatmentEnergy",
        "formula": "Water_Volume * Energy_Intensity / 1000"
      },
      "Chemical Emission": {
        "stub": "ChemicalEmission",
        "formula": "(Chemical_A * ChemA_EF + Chemical_B * ChemB_EF)"
      },
      "Sludge Emission": {
        "stub": "SludgeEmission",
        "formula": "Water_Volume * Sludge_Rate * Sludge_EF"
      },
      "Water CO2e": {
        "stub": "Water_CO2e",
        "formula": "TreatmentEnergy * Grid_EF_Water + ChemicalEmission + SludgeEmission"
      }
    },
    "data": {
      "Water_Volume": {
        "name": "Water Volume (cubic meters)",
        "value": 10000
      },
      "Energy_Intensity": {
        "name": "Energy Intensity (kWh/m³)",
        "value": 0.5
      },
      "Chemical_A": {
        "name": "Chemical A Usage (kg)",
        "value": 100
      },
      "ChemA_EF": {
        "name": "Chemical A Emission Factor (kg CO2e/kg)",
        "value": 2.5
      },
      "Chemical_B": {
        "name": "Chemical B Usage (kg)",
        "value": 50
      },
      "ChemB_EF": {
        "name": "Chemical B Emission Factor (kg CO2e/kg)",
        "value": 3.2
      },
      "Sludge_Rate": {
        "name": "Sludge Generation Rate (kg/m³)",
        "value": 0.02
      },
      "Sludge_EF": {
        "name": "Sludge Emission Factor (kg CO2e/kg)",
        "value": 0.8
      },
      "Grid_EF_Water": {
        "name": "Grid Emission Factor for Water Treatment (kg CO2e/kWh)",
        "value": 0.5
      }
    }
  },

  "Scope3_Total": {
    "_description": "Scope 3 Summary",
    "formulas": {
      "Scope 3 Total": {
        "stub": "Scope3_Total",
        "formula": "ScaledTransport_CO2e + Waste_CO2e + Water_CO2e"
      }
    },
    "data": {}
  },

  "GrandTotal": {
    "_description": "Organization Total Emissions",
    "formulas": {
      "Total Carbon Footprint": {
        "stub": "TotalCO2e",
        "formula": "Scope1_Total + Scope2_Total + Scope3_Total"
      },
      "Per Capita Emission": {
        "stub": "PerCapitaCO2e",
        "formula": "TotalCO2e / Employee_Count"
      },
      "Emission Intensity": {
        "stub": "EmissionIntensity",
        "formula": "TotalCO2e / Revenue * 1000000"
      }
    },
    "data": {
      "Employee_Count": {
        "name": "Number of Employees",
        "value": 500
      },
      "Revenue": {
        "name": "Annual Revenue (currency units)",
        "value": 50000000
      }
    }
  }
}

def get_model_template():
    return json.dumps(MODEL_TEMPLATE, indent=2, ensure_ascii=False)