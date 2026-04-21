# import yaml

# cfg_path = "./microdet.yml"
# with open(cfg_path, "r") as f:
#     cfg = yaml.safe_load(f)

# print(cfg["backbone"])
# print(cfg["neck"])
# print(cfg["head"])


import tomllib

cfg_path = "microdet.toml"
with open(cfg_path, "rb") as f:
    cfg = tomllib.load(f)
    model=cfg["model"]
    backbone=model["backbone"]
print("model: ",cfg["model"])
print("Backbone: ",backbone)
print("Neck: ", cfg["model"]["neck"])
print("Head: ",cfg["model"]["head"])

