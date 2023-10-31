# Import
import jax
from jax import numpy as jnp
from jax import random
from jax import grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
from flax.training.common_utils import shard
import optax
import cifar100_utils

# Load data
train_ds, test_ds = cifar100_utils.load_data()

# 数据预处理
def preprocess(x, y):
    x = jnp.float32(x) / 255.0
    y = jax.nn.one_hot(y, num_classes)
    return x, y

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# Define ResNet
class ResidualBlock(nn.Module):
    channels: int
    strides: Tuple[int, int] = (1, 1)

    def setup(self):
        self.conv1 = nn.Conv(channels=self.channels, kernel_size=(3, 3), strides=self.strides, padding="SAME")
        self.bn1 = nn.BatchNorm(use_running_average=False)
        self.conv2 = nn.Conv(channels=self.channels, kernel_size=(3, 3), padding="SAME")
        self.bn2 = nn.BatchNorm(use_running_average=False)
        self.shortcut = None
        if self.strides != (1, 1):
            self.shortcut = nn.Conv(channels=self.channels, kernel_size=(1, 1), strides=self.strides)

    def __call__(self, x, training=False):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = nn.relu(y)
        y = self.conv2(y)
        y = self.bn2(y, training=training)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return nn.relu(y + residual)

class ResNet(nn.Module):
    num_classes: int

    def setup(self):
        self.conv = nn.Conv(channels=64, kernel_size=(3, 3), padding="SAME")
        self.block1 = ResidualBlock(channels=64)
        self.block2 = ResidualBlock(channels=128, strides=(2, 2))
        self.block3 = ResidualBlock(channels=256, strides=(2, 2))
        self.pool = nn.avg_pool

        self.flatten = nn.Flatten()
        self.dense = nn.Dense(self.num_classes)

    def __call__(self, x, training=False):
        x = self.conv(x)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, (8, 8), strides=(1, 1), padding='VALID')
        x = self.flatten(x)
        x = self.dense(x)
        return x

# 创建ResNet模型实例
model = ResNet(num_classes=num_classes).initialize()

# Define train and evaluate 
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch["image"], training=True)
        loss = optax.softmax_cross_entropy(logits=logits, labels=batch["label"]).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics

@jax.jit
def eval_step(params, batch):
    logits = model.apply({"params": params}, batch["image"], training=False)
    loss = optax.softmax_cross_entropy(logits=logits, labels=batch["label"]).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(batch["label"], axis=-1))
    return loss, accuracy

# Train
# 定义训练参数
learning_rate = 0.001
weight_decay = 1e-5
batch_size = 128
num_epochs = 50
num_steps_per_epoch = len(train_ds) // batch_size

# 初始化优化器
optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).create(model.init_params)

# 初始化训练状态
rng = random.PRNGKey(0)
rng, key = random.split(rng)
dummy_batch = next(iter(train_ds))
params = model.init_params(key, dummy_batch["image"])

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)

# 训练循环
for epoch in range(num_epochs):
    rng, step_rng = random.split(rng)
    train_iter = iter(train_ds)

    epoch_metrics = []
    for step in range(num_steps_per_epoch):
        batch = next(train_iter)
        batch = shard(batch)

        state, metrics = train_step(state, batch)
        epoch_metrics.append(metrics)

        if step % 100 == 0:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Epoch {epoch}, Step {step}/{num_steps_per_epoch}, Metrics: {metrics_str}")

    epoch_metrics = jax.tree_map(jnp.mean, jax.tree_multimap(jnp.stack, *epoch_metrics))

    # 在验证集上评估模型
    eval_metrics = []
    for batch in test_ds:
        batch = shard(batch)
        loss, accuracy = eval_step(state.params, batch)
        eval_metrics.append({"loss": loss, "accuracy": accuracy})
    eval_metrics = jax.tree_map(jnp.mean, jax.tree_multimap(jnp.stack, *eval_metrics))

    # 打印训练和验证指标
    train_metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items())
    eval_metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
    print(f"Epoch {epoch} - Train metrics: {train_metrics_str} - Eval metrics: {eval_metrics_str}")
