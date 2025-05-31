import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Путь к логу TensorBoard
log_path = "tb_logs/EBSD_Euler"

# Загружаем данные из TensorBoard
event_acc = EventAccumulator(log_path)
event_acc.Reload()

# Получаем список всех скалярных тегов
tags = event_acc.Tags()["scalars"]
print("Доступные теги:", tags)

g_loss = event_acc.Scalars("losses/g_loss")
d_loss = event_acc.Scalars("losses/d_loss")
ae_loss1 = event_acc.Scalars("losses/ae_loss1")
ae_loss2 = event_acc.Scalars("losses/ae_loss2")
ae_loss = event_acc.Scalars("losses/ae_loss")

df = pd.DataFrame({
    "step": [x.step for x in g_loss],
    "generator_loss": [x.value for x in g_loss],
    "discriminator_loss": [x.value for x in d_loss],
    "adversarial_loss1": [x.value for x in ae_loss1],
    "adversarial_loss2": [x.value for x in ae_loss2],
    "adversarial_loss": [x.value for x in ae_loss],
})

plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["generator_loss"], label="Generator Loss")
plt.plot(df["step"], df["discriminator_loss"], label="Discriminator Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend(loc='center left', bbox_to_anchor=(0.75, 0.8))  # Смещаем легенду 
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["adversarial_loss1"], label="adversarial Loss 1")
plt.plot(df["step"], df["adversarial_loss2"], label="adversarial Loss 2")
plt.plot(df["step"], df["adversarial_loss"], label="adversarial Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid()
plt.show()