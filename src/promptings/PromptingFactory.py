from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy

from promptings.SCoder import SCoder
from promptings.SCoderWD import SCoderWD
from promptings.SCoderA import SCoderA
from promptings.SCoderWPV import SCoderWPV
from promptings.SCoderWPVD import SCoderWPVD
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
        elif prompting_name == "SCoderWD":
            return SCoderWD
        elif prompting_name == "SCoderWPV":
            return SCoderWPV
        elif prompting_name == "SCoderWPVD":
            return SCoderWPVD
        elif prompting_name == "SCoderA":
            return SCoderA
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
