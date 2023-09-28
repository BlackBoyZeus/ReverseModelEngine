import torch
import openai
from dataclasses import dataclass, field
from typing import List

@dataclass
class LayerInfo:
    name: str
    shape: tuple
    description: str = field(default="")

class UltraBrainGiftWrap:
    
    def __init__(self, model_path: str, api_key: str):
        self.model_path = model_path
        self.state_dict = self.load_state_dict_from_path()
        openai.api_key = api_key

    def load_state_dict_from_path(self) -> dict:
        """Load the state dictionary from the given model path."""
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        return state_dict

    @staticmethod
    def ask_gpt_about_layer(layer_name: str, layer_shape: tuple) -> str:
        """
        Use GPT to generate a description of what the layer might do based on its name and shape.
        """
        prompt = f"Based on the layer name '{layer_name}' and its shape {layer_shape}, what might this layer represent in a neural network architecture?"
        response = openai.Completion.create(engine="gpt-4.0-turbo", prompt=prompt, max_tokens=100)
        return response.choices[0].text.strip()

    def extract_info_from_state_dict(self) -> List[LayerInfo]:
        """Extract information and use GPT to understand each layer."""
        layer_infos = []
        
        for key, value in self.state_dict.items():
            layer_description = self.ask_gpt_about_layer(key, value.shape)
            layer_info = LayerInfo(name=key, shape=value.shape, description=layer_description)
            layer_infos.append(layer_info)
        
        return layer_infos

    @staticmethod
    def generate_instructions(layer_infos: List[LayerInfo]) -> str:
        """Generate instructions based on the extracted information and GPT's descriptions."""
        instructions = "Instructions to reverse engineer the model:\n\n"
        
        for layer_info in layer_infos:
            instructions += f"- Layer Name: {layer_info.name}\n"
            instructions += f"  Shape: {layer_info.shape}\n"
            instructions += f"  Description: {layer_info.description}\n"
            instructions += "\n"
        
        return instructions

    def process(self) -> str:
        """Main processing method to extract info and generate instructions."""
        layer_infos = self.extract_info_from_state_dict()
        return self.generate_instructions(layer_infos)

# Example usage:
api_key = 'YOUR_OPENAI_API_KEY'
model_path = 'path_to_your_model.pth'
wrapper = UltraBrainGiftWrap(model_path=model_path, api_key=api_key)
instructions = wrapper.process()
print(instructions)
