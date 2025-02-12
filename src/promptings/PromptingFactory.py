from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy
from promptings.MapCoder import MapCoder

from promptings.CodeSIM import CodeSIM
from promptings.variations.CodeSIMA import CodeSIMA
from promptings.variations.CodeSIMC import CodeSIMC
from promptings.variations.CodeSIMWD import CodeSIMWD
from promptings.variations.CodeSIMWPV import CodeSIMWPV
from promptings.variations.CodeSIMWPVD import CodeSIMWPVD

class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        elif prompting_name == "CodeSIM":
            return CodeSIM
        elif prompting_name == "CodeSIMA":
            return CodeSIMA
        elif prompting_name == "CodeSIMC":
            return CodeSIMC
        elif prompting_name == "CodeSIMWD":
            return CodeSIMWD
        elif prompting_name == "CodeSIMWPV":
            return CodeSIMWPV
        elif prompting_name == "CodeSIMWPVD":
            return CodeSIMWPVD
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
