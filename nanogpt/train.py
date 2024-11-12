import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from typing import Tuple
import optax
from functools import partial
 
from model import GPT2, GPT2Config

@struct.dataclass
class TrainConfig:
    """Training configuration."""
    # Model config
    model_config: GPT2Config
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    block_size: int = 128
    epochs: int = 5
    weight_decay: float = 0.1
    
    # Data config
    train_path: str = '../data/shakespeare/train.bin'
    val_path: str = '../data/shakespeare/val.bin'
    
    # Logging config
    log_every: int = 100
    eval_every: int = 500

    # Checkpoint config
    checkpoint: bool = False
    
    # TPU config
    num_devices: int = jax.device_count()
    
    def __post_init__(self):
        print(f"Initialized training config:")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Batch size: {self.batch_size} (per device: {self.batch_size//self.num_devices})")
        print(f"- Block size: {self.block_size}")
        print(f"- Epochs: {self.epochs}")
        print(f"- Weight decay: {self.weight_decay}")
        print(f"- Number of devices: {self.num_devices}")


# @partial(jax.jit, static_argnums=(3,))  # Comment this out temporarily
def train_step(train_state, batch, dropout_rng, model) -> Tuple[jnp.ndarray, TrainState]:
    """Train step for a single batch."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        X, Y = batch
        output = model.apply({'params': params}, X, rngs={'dropout': dropout_rng})
        # Extract logits from model output (assuming it's the first element)
        logits = output[0] if isinstance(output, tuple) else output
        
        logits = logits[:, :-1, :]
        Y = Y[:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
        return jnp.mean(loss)
    
    # Per device loss and gradient
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    # TODO Cant use while not pmamping grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Update parameters
    train_state = train_state.apply_gradients(grads=grads)
    
    return loss, train_state

@partial(jax.jit, static_argnums=(2,))
def eval_step(train_state, batch, model) -> jnp.ndarray:
    """Evaluation step."""
    x, y = batch
    output = model.apply({'params': train_state.params}, x, train=False)
    logits = output[0] if isinstance(output, tuple) else output
    
    # Shift logits and targets
    logits = logits[:, :-1, :]
    targets = y[:, 1:]
    
    # Compute loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)

def evaluate(key, train_state, val_ds, model, config) -> jnp.ndarray:
    """Evaluate the model on the validation set."""
    print("\nStarting evaluation...")
    total_loss = 0
    num_batches = len(val_ds) // (config.batch_size * config.block_size)
    
    for i in range(num_batches):
        key, batch_key = jax.random.split(key)
        batch = get_batch(batch_key, val_ds, config.batch_size, config.block_size)
        loss = eval_step(train_state, batch, model)
        total_loss += loss
        
        if i % 10 == 0:
            print(f"Eval batch {i}/{num_batches}, loss: {loss:.4f}")
    
    return total_loss / num_batches

def count_params(params: FrozenDict) -> int:
    """Count the number of parameters in the model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def param_decay(params: FrozenDict) -> FrozenDict:
    """Compute the parameter decay mask for non-bias parameters."""
    def _is_kernel(path, _):
        return path[-1] == 'kernel'
    
    flat_mask = jax.tree.map(
        lambda path, _: 1.0 if _is_kernel(path, None) else 0.0,
        jax.tree_util.tree_leaves(params),
        params
    )
    return FrozenDict(flat_mask)

def init_train_state(key, config: TrainConfig, model: GPT2, input_shape: Tuple[int, int]) -> TrainState:
    """Initialize the training state."""
    print("\nInitializing training state...")
    
    if config.checkpoint:
        pass
    else:
        # Create dummy input
        #dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
        
        # Initialize parameters
        model = GPT2(config.model_config)
        params = model.init(key)
        params = params['params']

    # Create optimizer
    #decay_mask = param_decay(params)
    tx = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        #mask=decay_mask
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def get_batch(key, data, batch_size, block_size):
    """Get a random batch of data."""
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5) 
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    args = parser.parse_args()

    # Initialize random key
    key = jax.random.PRNGKey(0)
    key, init_key, dropout_key = jax.random.split(key, 3)

    # Load data
    print("Loading data...")
    with open(TrainConfig.train_path, 'rb') as f:
        train_data = jnp.frombuffer(f.read(), dtype=jnp.uint8)
    with open(TrainConfig.val_path, 'rb') as f:
        val_data = jnp.frombuffer(f.read(), dtype=jnp.uint8)
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    # Initialize model and training state
    print("Initializing model...")
    config = GPT2Config()
    model = GPT2(config)

    # Initialize training state
    input_shape = (args.batch_size, args.block_size)
    train_config = TrainConfig(
        model_config=config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size
    )
    state = init_train_state(init_key, train_config, model, input_shape)
    print(f"Number of parameters: {count_params(state.params):,}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Training
        total_loss = 0
        num_batches = len(train_data) // (args.batch_size * args.block_size)
        
        for step in range(num_batches):
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{num_batches}")
            
            key, batch_key = jax.random.split(key)  # Get a new key for this batch
            batch = get_batch(batch_key, train_data, args.batch_size, args.block_size)
            dropout_key, new_dropout_key = jax.random.split(dropout_key)
            loss, state = train_step(state, batch, new_dropout_key, model)
            total_loss += loss

            if step % 100 == 0:
                print(f"Training loss: {loss:.4f}")

        avg_train_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} average training loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = evaluate(key, state, val_data, model, train_config)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

    print("\nTraining complete!")