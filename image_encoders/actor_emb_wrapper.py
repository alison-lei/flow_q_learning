import pickle as pkl

class ActorEmbWrapper(nn.Module):
    use_prioprio: bool = True
    proprio_latent_dim: int = 64

    # gets rid of the need of the need for a separate method to define the parameters and submodules
    @nn.compact
    


