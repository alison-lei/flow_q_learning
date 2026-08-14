import flax.linen as nn
import jax.numpy as jnp

from image_encoder.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs

IMAGE_KEYS = ("side_policy_256", "wrist_1")

class CriticEmbWrapper(nn.Module):
    use_proprio: bool = True
    proprio_latent_dim: int = 64
    

    @nn.compact
    def __call__(self, observations, train=True):
        
        # shared resnet cnn backbone across both cameras, merge embeddings from both cameras
        backbone = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True, name = "pretrained_encoder", 
        )

        image_embeds = []

        for image_key in IMAGE_KEYS:
            image = observations[image_key]
            # assumes window_size > 1
            if image.shape[-4] > 1:
                image = image[..., -1, :, :, :]

            # flushed out encoder that is backbone + trainable pooling head + bottleneck of Dense->LayerNorm->tanh
            encoder = PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=backbone,
                name=f"encoder_{image_key}",
            )
            image_embeds.append(encoder(image, train=train))

        encoded = jnp.concatenate(image_embeds, axis=-1)

        if self.use_proprio:
            state = observations["state"]
            state = state.reshape(state.shape[0], -1)
            state = nn.Dense(self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform())(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)
        return encoded


class ActorEmbWrapper(nn.Module):
    use_proprio: bool = True
    proprio_latent_dim: int = 64

    # gets rid of the need of the need for a separate method to define the parameters and submodules
    @nn.compact
    # forward call
    def __call__(self, observations, train=True):
        encoded = observations["embeddings"]
        if self.use_proprio:
            state = observations["state"]
            state = state.reshape(state.shape[0], -1)
            state = nn.Dense(self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform())(state) # kernel is the same as weight matrix
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)
        return encoded



