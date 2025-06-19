import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from models import DynamicVWAPTransformer
from losses import vwap_afs_loss

# === Paramètres ===
lookback = 10
n_ahead = 5
num_features = 6
lambda_impact = 0.01
eta_impact = 0.01

# === Données factices ===
x_train = np.random.rand(128, lookback + n_ahead - 1, num_features).astype(np.float32)
y_train = np.random.rand(128, n_ahead, 2).astype(np.float32)

x_val = np.random.rand(32, lookback + n_ahead - 1, num_features).astype(np.float32)
y_val = np.random.rand(32, n_ahead, 2).astype(np.float32)

# === Initialisation du modèle ===
model = DynamicVWAPTransformer(
    lookback=lookback,
    n_ahead=n_ahead,
    hidden_size=64,
    hidden_rnn_layer=2,
    num_heads=2,
    num_embedding=4
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=lambda y_true, y_pred: vwap_afs_loss(y_true, y_pred, lambda_impact, eta_impact)
)

# === Entraînement ===
model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# === Sauvegarde ===
model.save("trained_vwap_model.keras")