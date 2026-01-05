import torch
from models.deepfm import DeepFM
from preprocess import MovieLensDataLoader

loader = MovieLensDataLoader("data/raw/ml-1m")
_, field_dims = loader.load_and_preprocess();

model = DeepFM(field_dims, embed_dim=16)
model.load_state_dict(torch.load("deepfm_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.tensor([[1, 100]], dtype=torch.long)

traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("deepfm_traced.pt")
print("Move exported successfully as deepfm_traced.pt")