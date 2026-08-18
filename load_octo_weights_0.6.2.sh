# --- Clone Octo and install without letting it pin old deps ---
git clone https://github.com/octo-models/octo.git
cd octo
pip install -e . --no-deps

# --- Missing runtime deps (safe, no jax involvement) ---
pip install astunparse tf_keras huggingface_hub
pip install "transformers<5.0"                 # v5.0 deleted all Flax model classes

# --- TensorFlow + pinned jax/jaxlib/ml-dtypes together, so TF can't drag jax down ---
pip install "tensorflow==2.21.0" "jax==0.6.2" "jaxlib==0.6.2" "ml-dtypes>=0.5.0"

# --- Stub out dlimp (Octo's training-data pipeline lib — not needed for inference,
#     and the real one drags ml-dtypes back down to an incompatible version) ---
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/dlimp.py" << 'EOF'
# stub: real dlimp is a TF-datasets pipeline lib, not needed for inference-only use.
class _Dummy:
    def __class_getitem__(cls, item):
        return cls
    def __call__(self, *args, **kwargs):
        return self

def __getattr__(name):
    return _Dummy
EOF

# --- Patch removed/renamed jax APIs in Octo's source (editable install, so this sticks) ---
sed -i 's/PRNGKey = jax\.random\.KeyArray/PRNGKey = jax.Array/' octo/utils/typing.py
grep -rl "jax\.tree_map(" octo/ | xargs sed -i 's/jax\.tree_map(/jax.tree_util.tree_map(/g'

# --- Patch orbax's Python 3.10 asyncio bug (unrelated to any package version) ---
ORBAX_LIMITS=$(python -c "import orbax.checkpoint._src.serialization.limits as m; print(m.__file__)")
sed -i 's/self\._cv = asyncio\.Condition(lock=asyncio\.Lock())/self._cv = asyncio.Condition()/' "$ORBAX_LIMITS"

# --- Verify everything ---
python -c "import jax; print('jax', jax.__version__)"
python -c "from octo.model.octo_model import OctoModel; OctoModel.load_pretrained('hf://rail-berkeley/octo-small'); print('octo ok')"
python -c "import flax, orbax.checkpoint, optax; print('fql ok')"