#include <torch/script.h>
#include <iostream>
#include <memory>

int main() {
    // Load Model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("deepfm_traced.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "DeepFM model loaded successfully in C++!!!\n";
    // Supposed we are predicting userId = 0, movieId = 10
    std::vector<int64_t> inputs = {0, 10};
    at::Tensor tensor_input = torch::from_blob(inputs.data(), {1, 2}, torch::kInt64);

    // Process forward calculate
    at::Tensor output = module.forward({tensor_input}).toTensor();
    std::cout << "Prediction score: " << output.item<float>() << std::endl;

    return 0;
}