import torch

b = torch.load(r"C:\Users\shadb\Downloads\dataset\deployment\potato_bundle.pth",
               map_location="cpu", weights_only=False)
print("FORMAT:", b["format"])
print("CLASSES:", b["class_names"])
for m in b["models"]:
    n = sum(v.numel() for v in m["state_dict"].values())
    print(f"  {m['key']:20s} {m['timm_name']:42s} img={m['img_size']} params={n/1e6:.1f}M OK")
print("BUNDLE VERIFIED - LOADABLE")
