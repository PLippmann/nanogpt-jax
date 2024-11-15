import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from typing import Tuple
import numpy as np
import flax.jax_utils
import argparse
from model import GPT2, GPT2Config
from functools import partial
import wandb

@struct.dataclass
class TrainConfig:
    """Training configuration."""
    # Model config
    model_config: GPT2Config

    # Random seed
    seed: int = 1337
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 16 # Per device
    weight_decay: float = 0.1
    train_steps: int = 200000
    val_steps: int = 100
    
    # Choose data to be used [openwebtext, shakespeare]
    data_set: str = 'openwebtext'
    
    # GCP data source config
    tpu: bool = False
    bucket_name: str = 'nano-openwebtext'
    
    # Data config
    @property
    def data_dir(self) -> str:
        if self.tpu:
            return f'gs://{self.bucket_name}'
        return f'../data/{self.data_set}'
    
    @property
    def train_path(self) -> str:
        return f'{self.data_dir}/train.bin'
    
    @property
    def val_path(self) -> str:
        return f'{self.data_dir}/val.bin'
    
    # Evaluation config
    log_every: int = 100 # Interval for logging training loss
    eval_every: int = 1000 # Interval for logging validation loss

    # Checkpoint read/write config
    checkpoint: bool = False
    output_dir: str = 'out' # TODO implement checkpointing
    
    # TPU config
    num_devices: int = jax.device_count()
    
    @property
    def global_batch_size(self) -> int:
        """Total batch size across all devices"""
        return self.batch_size * self.num_devices
    
    @property
    def per_device_batch_size(self) -> int:
        """Batch size per device"""
        return self.batch_size // self.num_devices
    
    def __post_init__(self):
        print(f"Initialized training config:")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Global batch size: {self.global_batch_size}")
        print(f"- Per device batch size: {self.per_device_batch_size}")
        print(f"- Weight decay: {self.weight_decay}")
        print(f"- Number of devices: {self.num_devices}")


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))  # Add donate_argnums for memory efficiency
def train_step(train_state, batch, dropout_rng) -> Tuple[jnp.ndarray, TrainState]:
    """Train step for a single batch."""
    dropout_rng, new_dropout_key = jax.random.split(dropout_rng)

    def loss_fn(params):
        X, Y = batch
        output = train_state.apply_fn({'params': params}, X, rngs={'dropout': dropout_rng})
        logits = output[0] if isinstance(output, tuple) else output
        
        # Shift targets to the right
        logits = logits[:, :-1, :]
        Y = Y[:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
        return jnp.mean(loss)
    
    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    grads = jax.lax.pmean(grads, axis_name='batch') # Average gradients across devices
    loss = jax.lax.pmean(loss, axis_name='batch') # Average loss across devices
    
    train_state = train_state.apply_gradients(grads=grads)
    return loss, train_state

@partial(jax.pmap, axis_name='batch')
def eval_step(train_state, batch) -> jnp.ndarray:
    """Evaluation step."""
    x, y = batch
    output = train_state.apply_fn({'params': train_state.params}, x)
    logits = output[0] if isinstance(output, tuple) else output
    
    # Shift targets to the right
    logits = logits[:, :-1, :]
    targets = y[:, 1:]
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jax.lax.pmean(jnp.mean(loss), axis_name='batch')

def evaluate(key, train_state, val_ds, model, config) -> jnp.ndarray:
    """Evaluate the model on the validation set."""
    print("\nStarting evaluation...")
    total_loss = 0
    num_batches = len(val_ds) // (config.batch_size * config.block_size)
    
    for i in range(num_batches):
        key, batch_key = jax.random.split(key)
        batch = get_batch(batch_key, val_ds, config.batch_size, config.block_size)
        loss = eval_step(train_state, batch)
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
        pass # TODO implement checkpointing
    else:
        # Initialize params
        model = GPT2(config.model_config)
        params = model.init(key)
        params = params['params']

    # Create optimizer
    #decay_mask = param_decay(params) TODO implement weight decay
    tx = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        #mask=decay_mask TODO implement weight decay
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def get_batch(key, data, batch_size, block_size):
    """Get a random batch of data."""
    data_len = len(data)
    adjusted_max_start_idx = min(data_len - block_size, np.iinfo(np.int32).max)
    
    ix = jax.random.randint(
        key, 
        (batch_size,), 
        0, 
        adjusted_max_start_idx,
        dtype=jnp.int32  # Use int32 to align with JAX's internal handling
    )
    
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def load_data(data_path: str) -> np.ndarray:
    """Load data from binary file."""
    if data_path.startswith('gs://'): # GCP storage bucket
        import tensorflow as tf
        data = tf.io.gfile.GFile(data_path, 'rb')
        return np.frombuffer(data.read(), dtype=np.uint16)
    else:
        return np.memmap(data_path, dtype=np.uint16, mode='r')

if __name__ == "__main__":
    # Model config
    config = GPT2Config()
    
    # Create initial config with defaults
    train_config = TrainConfig(model_config=config)

    # Parse command line arguments wåith defaults from config
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=train_config.batch_size)
    parser.add_argument('--learning_rate', type=float, default=train_config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=train_config.weight_decay)
    parser.add_argument('--tpu', action='store_true', default=train_config.tpu, help='Use TPU and load from GCS bucket')
    parser.add_argument('--wandb_project', type=str, default='nanogpt', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name')
    args = parser.parse_args()

    # Update config with parsed arguments
    train_config = train_config.replace(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tpu=args.tpu
    )

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "learning_rate": train_config.learning_rate,
            "batch_size": train_config.batch_size,
            "weight_decay": train_config.weight_decay,
            "model_config": train_config.model_config.__dict__,
            "dataset": train_config.data_set,
        }
    )

    # Initialize random key
    key = jax.random.PRNGKey(train_config.seed)
    key, init_key, dropout_key = jax.random.split(key, 3)

    # Load training data
    train_data = load_data(train_config.train_path)

    # Load validation data
    val_data = load_data(train_config.val_path)

    print("Loaded data...")
    print(f"Train data size: {len(train_data):,} tokens")
    print(f"Val data size: {len(val_data):,} tokens")

    # Initialize model and training state
    print("Initializing model...")
    model = GPT2(config)

    # Initialize training state
    input_shape = (train_config.per_device_batch_size, config.block_size)
    state = init_train_state(init_key, train_config, model, input_shape)
    num_params = count_params(state.params)
    print(f"Number of parameters: {num_params:,}")
    wandb.run.summary["num_parameters"] = num_params

    # Replicate state across devices
    state = flax.jax_utils.replicate(state)
    
    # Create mesh-replicated random keys
    dropout_keys = jax.random.split(dropout_key, train_config.num_devices)
    dropout_key = jnp.array(dropout_keys)

    # Training loop
    print("\nStarting training...")
    global_step = 0
    
    # Training
    total_loss = 0
    
    for step in range(train_config.train_steps):
        if step % train_config.log_every == 0:
            print(f"Step {step}/{train_config.train_steps}")
        
        # Create per-device batches for training
        key, batch_key = jax.random.split(key)
        batch_keys = jax.random.split(batch_key, train_config.num_devices)
        batches = [
            get_batch(k, train_data, train_config.per_device_batch_size, config.block_size)
            for k in batch_keys
        ]
        # Stack for pmap
        batch = jax.tree_map(lambda *x: jnp.stack(x), *batches)
        
        # Update dropout keys
        dropout_keys = jax.random.split(dropout_key[0], train_config.num_devices)
        dropout_key = jnp.array(dropout_keys)
        
        loss, state = train_step(state, batch, dropout_key)
        loss_value = jnp.mean(loss)
        total_loss += loss_value
        global_step += 1

        # Log training metrics
        wandb.log({
            "train/loss": loss_value,
            "train/learning_rate": train_config.learning_rate,
            "train/step": step,
            "train/global_step": global_step,
        }, step=global_step)

        if step % 100 == 0:
            print(f"Training loss: {loss_value:.4f}")
            
        # Validation during training
        if step % train_config.eval_every == 0:
            val_total_loss = 0
            num_val_batches = min(10, train_config.val_steps)  # Limit validation batches during training
            
            for val_step in range(num_val_batches):
                key, val_batch_key = jax.random.split(key)
                val_batch_keys = jax.random.split(val_batch_key, train_config.num_devices)
                val_batches = [
                    get_batch(k, val_data, train_config.per_device_batch_size, config.block_size)
                    for k in val_batch_keys
                ]
                val_batch = jax.tree_map(lambda *x: jnp.stack(x), *val_batches)
                
                val_loss = eval_step(state, val_batch)
                val_loss_value = jnp.mean(val_loss)
                val_total_loss += val_loss_value
            
            avg_val_loss = val_total_loss / num_val_batches
            print(f"Validation loss at step {step}: {avg_val_loss:.4f}")
            wandb.log({
                "val/loss": avg_val_loss,
                "val/step": step,
            }, step=global_step)

    avg_train_loss = total_loss / train_config.train_steps
    print(f"\nAverage training loss: {avg_train_loss:.4f}")
    wandb.log({
        "train/final_loss": avg_train_loss,
    }, step=global_step)

    # Final full validation
    val_total_loss = 0
    
    for step in range(train_config.val_steps):
        key, batch_key = jax.random.split(key)
        batch_keys = jax.random.split(batch_key, train_config.num_devices)
        batches = [
            get_batch(k, val_data, train_config.per_device_batch_size, config.block_size)
            for k in batch_keys
        ]
        batch = jax.tree_map(lambda *x: jnp.stack(x), *batches)
        
        val_loss = eval_step(state, batch)
        val_loss_value = jnp.mean(val_loss)
        val_total_loss += val_loss_value
        
    avg_val_loss = val_total_loss / train_config.val_steps
    print(f"Final validation loss: {avg_val_loss:.4f}")
    wandb.log({
        "val/final_loss": avg_val_loss,
    }, step=global_step)

    print("\nTraining complete!")
    wandb.finish()