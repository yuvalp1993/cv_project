from xcpetion import build_xception_backbone
from utils import get_nof_params
from models import get_xception_based_model


original_xception = build_xception_backbone(pretrained = True)
original_params = get_nof_params(original_xception)

print(f"Original Xception Parameters: {original_params}")

modified_xception = get_xception_based_model()
modified_params = get_nof_params(modified_xception)
print(f"Modified Xception Parameters: {modified_params}")

added_params = modified_params - original_params
print(f"Parameters Added with MLP: {added_params}")