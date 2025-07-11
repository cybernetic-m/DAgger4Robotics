try: #import working in colab
    from DAgger4Robotics.model.SimplePolicyNet import SimplePolicyNet
    from DAgger4Robotics.model.DeepPolicyNet import DeepPolicyNet
except ModuleNotFoundError:
    # fallback when executed in local
    from model.SimplePolicyNet import SimplePolicyNet
    from model.DeepPolicyNet import DeepPolicyNet

class NetworkInterface:
    def __init__(self, net_type, input_dim, output_dim):
        if net_type == "simple":
            self.model = SimplePolicyNet(input_dim, output_dim)
        elif net_type == "deep":
            self.model = DeepPolicyNet(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported net type: {net_type}")

    def get_model(self):
        return self.model
    
    def summary(self):
        return print(self.model)