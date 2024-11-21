import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from typing import Tuple, Callable
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
    learning_rate: float = 6e-4
    batch_size: int = 16 # Per device
    weight_decay: float = 1e-1
    train_steps: int = 300000
    val_steps: int = 100
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    adam_eps: float = 1e-5 # This alledgedly helps with convergence

    # Learning rate schedule config
    init_lr: float = learning_rate * 0.1
    peak_lr: float = learning_rate
    warmup_steps: int = 2000
    decay_steps: int = train_steps * 0.8
    end_lr: float = learning_rate * 0.1
    
    # Choose data to be used [openwebtext, shakespeare]
    data_set: str = 'openwebtext'
    
    # GCP data source config
    tpu: bool = True
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
        return self.batch_size * self.num_devices * self.gradient_accumulation_steps
    
    @property
    def per_device_batch_size(self) -> int:
        """Batch size per device"""
        return self.batch_size
    
    def __post_init__(self):
        print(f"Initialized training config:")
        print(f"- Global batch size: {self.global_batch_size}")
        print(f"- Per device batch size: {self.per_device_batch_size}")
        print(f"- Weight decay: {self.weight_decay}")
        print(f"- Number of devices: {self.num_devices}")

def create_learning_rate_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float
) -> Callable[[int], float]:
    """Creates a learning rate schedule with linear warmup and cosine decay."""
    
    def schedule(step: int) -> float:
        # Linear warmup
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        warmup_lr = init_value + (peak_value - init_value) * warmup_factor
        
        # Cosine decay
        decay_progress = jnp.maximum(0.0, (step - warmup_steps) / decay_steps)
        decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(decay_progress, 1.0)))
        decay_lr = end_value + (peak_value - end_value) * decay_factor
        
        return jnp.where(step < warmup_steps, warmup_lr, decay_lr)
    
    return schedule

@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(train_state, batch, dropout_rng, train_step_num) -> Tuple[jnp.ndarray, TrainState, jnp.ndarray]:
    """Train step for a single batch."""
    dropout_rng, new_dropout_key = jax.random.split(dropout_rng)

    def loss_fn(params):
        X, Y = batch
        output = train_state.apply_fn({'params': params}, X, rngs={'dropout': dropout_rng})
        logits = output[0] if isinstance(output, tuple) else output
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Calculate gradient norm before clipping
    grad_norm = optax.global_norm(grads)
    
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    train_state = train_state.apply_gradients(grads=grads)
    return loss, train_state, grad_norm

@partial(jax.pmap, axis_name='batch')
def eval_step(train_state, batch) -> jnp.ndarray:
    """Evaluation step."""
    x, y = batch
    output = train_state.apply_fn({'params': train_state.params}, x)
    logits = output[0] if isinstance(output, tuple) else output
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
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

    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule(
        init_value=config.init_lr,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.end_lr
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=config.weight_decay,
            #mask=param_decay_mask(gpt_params), #TODO implement param decay
        ),
    )
    tx = optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    ), lr_schedule  # Return the schedule along with the state

def get_batch(key, data, batch_size, block_size):
    """Get a random batch of data."""
    data_len = len(data)
    # Adjust the maximum start index to ensure we always have enough tokens
    adjusted_max_start_idx = data_len - block_size - 1
    
    if adjusted_max_start_idx <= 0:
        raise ValueError(f"Data length ({data_len}) must be greater than block_size + 1 ({block_size + 1})")
    
    # Generate random floats in [0, 1) and scale them to our desired range
    random_floats = jax.random.uniform(key, (batch_size,))
    ix = (random_floats * adjusted_max_start_idx).astype(jnp.int32)
    
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

def check_data_validity(data):
    """Check for NaN or infinite values in the data."""
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

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
    check_data_validity(train_data)

    # Load validation data
    val_data = load_data(train_config.val_path)
    check_data_validity(val_data)

    print("Loaded data...")
    print(f"Train data size: {len(train_data):,} tokens")
    print(f"Val data size: {len(val_data):,} tokens")

    # Initialize model and training state
    print("Initializing model...")
    model = GPT2(config)

    # Initialize training state
    input_shape = (train_config.per_device_batch_size, config.block_size)
    state, lr_schedule = init_train_state(init_key, train_config, model, input_shape)  # Get lr_schedule
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
    total_loss = 0
    
    # Create replicated step counter for pmap
    step_counter = jnp.zeros((train_config.num_devices,), dtype=jnp.int32)
    
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
        batch = jax.tree.map(lambda *x: jnp.stack(x), *batches)
        
        # Update dropout keys
        dropout_keys = jax.random.split(dropout_key[0], train_config.num_devices)
        dropout_key = jnp.array(dropout_keys)
        
        loss, state, grad_norm = train_step(state, batch, dropout_key, step_counter)
        loss_value = jnp.mean(loss)
        grad_norm_value = jnp.mean(grad_norm)
        
        total_loss += loss_value
        global_step += 1

        # Log training metrics
        if step % train_config.log_every == 0:
            current_lr = lr_schedule(step)  # Get learning rate for logging
            print(f"Step {step}, Loss: {loss_value:.4f}, LR: {current_lr:.6f}, Grad norm: {grad_norm_value:.4f}")
            
            wandb.log({
                "train/loss": loss_value,
                "train/learning_rate": current_lr,
                "train/grad_norm": grad_norm_value,
                "train/step": step,
                "train/global_step": global_step,
            }, step=global_step)

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
                val_batch = jax.tree.map(lambda *x: jnp.stack(x), *val_batches)
                
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
        batch = jax.tree.map(lambda *x: jnp.stack(x), *batches)
        
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