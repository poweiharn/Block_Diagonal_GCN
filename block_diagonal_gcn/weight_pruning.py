import torch
import torch.nn.utils.prune as prune

# Threshold pruning
# Return mask M for the weight matrix
# m_ij = 0 if w_ij < threshold
# m_ij = 1, otherwise
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold



def main():
    # Example
    weight_example = [[0.3, -0.2], [0.1, -0.4]]
    x_data = torch.tensor(weight_example)
    # Print out the mask
    print(torch.abs(x_data) > 0.3)


if __name__ == "__main__":
    main()