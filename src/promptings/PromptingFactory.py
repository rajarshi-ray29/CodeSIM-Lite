from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy

from promptings.SCoder import SCoder
from promptings.SCoderV16 import SCoderV16
from promptings.MapCoder import MapCoder


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
        elif prompting_name == "SCoder":
            return SCoder
        elif prompting_name == "SCoderV16":
            return SCoderV16
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
