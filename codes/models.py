import pandas as pd
import numpy as np
import copy
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Input, Concatenate, Flatten, Reshape, ReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras import backend as K

# --------------------------------------------------------
# Automatic device selection: use GPU if available, else CPU
# --------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Allow dynamic memory growth instead of grabbing all memory
            tf.config.experimental.set_memory_growth(gpu, True)
        _DEFAULT_DEVICE = '/GPU:0'
        print(f"[INFO] Using GPU: {_DEFAULT_DEVICE}")
    except RuntimeError as e:
        print(f"[WARN] Could not set GPU memory growth: {e}")
        _DEFAULT_DEVICE = '/GPU:0'
else:
    _DEFAULT_DEVICE = '/CPU:0'
    print("[INFO] No GPU found, using CPU.")


def convert_all_datetime_to_float(df, unit="D"):
    """
    Convert all columns containing timestamps (datetime dtype or mixed with timestamps)
    into float values (days/seconds/hours since epoch).
    
    Args:
        df   : pandas DataFrame
        unit : "D"=days, "h"=hours, "s"=seconds
    
    Returns:
        DataFrame with datetime-like columns converted to float64
    """
    out = df.copy()
    
    # Loop over columns
    for col in out.columns:
        series = out[col]
        
        # Case 1: column already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            if unit == "s":
                out[col] = series.astype("int64") / 1e9
            elif unit == "h":
                out[col] = series.astype("int64") / 1e9 / 3600.0
            elif unit == "D":
                out[col] = series.astype("int64") / 1e9 / 86400.0
            continue
        
        # Case 2: object column → check if it contains any timestamps
        if series.dtype == object:
            # Try parsing values that look like datetimes
            def _try_convert(val):
                if isinstance(val, (pd.Timestamp, np.datetime64)):
                    v = pd.to_datetime(val)
                    if unit == "s":
                        return v.value / 1e9
                    elif unit == "h":
                        return v.value / 1e9 / 3600.0
                    elif unit == "D":
                        return v.value / 1e9 / 86400.0
                return np.nan  # keep as NaN if not a timestamp
            
            if any(isinstance(v, (pd.Timestamp, np.datetime64)) for v in series.dropna()):
                out[col] = series.map(
                    lambda v: _try_convert(v) if pd.notna(v) else np.nan
                ).astype(float)
    
    return out


def create_dataset(data_copy, ba_copy, seq_length, mapping_dict, 
                   key='SCCRIP_ID', 
                   target='Hgb_F_hemoglobineid',
                   addLabel = None,
                   hasHuStatus=True, 
                   keepTarget=True):
    sequences = []
    labels = []
    seq_ba = []
    date = []
    y_target = []
    g = data_copy.groupby(key)
    for k in g.groups.keys():
        t = g.get_group(k)
        t = t.sort_index()
        t = t.fillna(0)
        if len(t) > seq_length:
            date.append(list(t['date']))
            t_ = copy.deepcopy(t)
            t_["date"] = t_["date"].astype(str)
            if hasHuStatus:
                if addLabel:
                    labels.append(np.array(t_[['HU_status','Id_encoded','Ages', 'date', addLabel]]))
                else:                                        
                    labels.append(np.array(t_[['HU_status','Id_encoded','Ages', 'date']]))
            else:
                if addLabel:
                    labels.append(np.array(t_[['Id_encoded','Ages', 'date', addLabel]]))
                else:
                    labels.append(np.array(t_[['Id_encoded','Ages', 'date']]))
         
            t.drop([key], inplace=True, axis=1)
            if hasHuStatus:
                if addLabel:
                    t.drop(['HU_status','Id_encoded', 'date', addLabel], inplace=True, axis=1)
                else:
                    t.drop(['HU_status','Id_encoded', 'date'], inplace=True, axis=1)
            else:
                if addLabel:
                    t.drop(['Id_encoded', 'date', addLabel], inplace=True, axis=1)
                else:
                    t.drop(['Id_encoded', 'date'], inplace=True, axis=1)
            if not keepTarget:
                t.drop(target, inplace=True, axis=1)

            t = convert_all_datetime_to_float(t, unit="D")
            t = t.fillna(0)
            sequences.append(t.to_numpy())
            
            t_ba = ba_copy[ba_copy['Id_encoded'] == mapping_dict[k]]
            t_ba.drop('Id_encoded', inplace=True, axis=1)
            t_ba = convert_all_datetime_to_float(t_ba, unit="D")
            seq_ba.append(t_ba.to_numpy())
            
            y_target.append(np.array(t_[target]))
    return sequences, seq_ba, labels, date, y_target
    
    
def shuffle_dataset(sequences, seq_ba, labels, date, y_target):
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    sequences_copy = [sequences[i] for i in indices]
    labels_copy = [labels[i] for i in indices]
    seq_ba_copy = [seq_ba[i] for i in indices]
    date_copy =  [date[i] for i in indices]
    y_target_copy = [y_target[i] for i in indices]
    return sequences_copy, seq_ba_copy, labels_copy, date_copy, y_target_copy


def create_sequences_with_background(data, labels, backgrounds, date, y_target, seq_length):
    X, y, seq_map, X_bg, dt, ls = [], [], [], [], [], []
    for seq_idx, (seq, label_seq, bg, time, y_t) in enumerate(zip(data, labels, backgrounds, date, y_target)):
        for i in range(len(seq) - seq_length):
            X.append(seq[i:i + seq_length])  # All features except the last column (target)
            y.append(y_t[i + seq_length])  # Only the last column (target)
            ls.append(label_seq[i + seq_length])
            X_bg.append(bg)
            delta = (time[i + seq_length] - time[i + seq_length -1]) / pd.Timedelta(days=1)
            # print(delta)
            dt.append(delta)
            seq_map.append(seq_idx)  # Keep track of which original sequence each example came from
            
    X = np.asarray(X).astype('float32')
    X_bg = np.asarray(X_bg).astype('float32')
    y = np.asarray(y).astype('float32')
    dt = np.asarray(dt).astype('float32')
    return X, y, X_bg, dt, ls, np.array(seq_map)


def normalize_X(X, X_bg):
    # Flatten the data for normalization
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_flat_normalized = scaler.fit_transform(X_flat)
    X_normalized = X_flat_normalized.reshape(n_samples, n_timesteps, n_features)
    bg_scaler = StandardScaler()
    X_bg_normalized = bg_scaler.fit_transform(X_bg.reshape(-1, X_bg.shape[-1])).reshape(X_bg.shape)
    
    return X_normalized, X_bg_normalized


class CombinedLSTMModel:
    def __init__(self, seq_length, n_features, bg_features, summary=True):
        self.seq_length = int(seq_length)
        self.n_features = int(n_features)
        self.bg_features = int(bg_features)
        self.summary = summary
        self.model = self.build_model()
    
    def build_model(self):
        # Build the model on the selected device (GPU if available, else CPU)
        with tf.device(_DEFAULT_DEVICE):
            # Inputs
            input_seq = Input(shape=(self.seq_length, self.n_features), name="input_seq")
            input_bg  = Input(shape=(1, self.bg_features), name="input_bg")
            input_dt  = Input(shape=(1, 1), name="input_dt")
        
            # bg + dt
            bg_concat = Concatenate(axis=-1, name="bg_plus_dt")([input_bg, input_dt])   # (1, bg_features+1)
            bg_flat   = Flatten(name="bg_flat")(bg_concat)
            bg_feat_total = int(self.bg_features) + 1  # include dt
        
            # LSTM: MUST be 2*(bg_features+1) so the reshape to (3, bg_feat_total) works
            lstm_units = 2 * bg_feat_total
            lstm_out = LSTM(lstm_units, name="seq_lstm")(input_seq)
        
            # fuse
            concatenated = Concatenate(name="fuse_seq_bg")([lstm_out, bg_flat])  # length = 3*bg_feat_total
        
            # sanity check at runtime (won't run in graph mode, but helpful when eager)
            if tf.executing_eagerly():
                concat_dim = concatenated.shape[-1]
                expected = 3 * bg_feat_total
                assert concat_dim == expected, f"Concat dim {concat_dim} != {expected} (3*(bg_features+1))"
        
            # reshape to (3, bg_feat_total) so Conv1D with kernel_size=3 fits exactly
            reshaped = Reshape((3, bg_feat_total), name="reshape_to_conv")(concatenated)
        
            conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', name="post_concat_conv")(reshaped)
            fla   = Flatten(name="flat_after_conv")(conv2)
        
            den1 = Dense(128, name="dense_128")(fla)
            den2 = Dense(16,  name="dense_16")(den1)
            den  = Dense(1,   name="dense_out")(den2)
            output = ReLU(name="relu_out")(den)
        
            model = Model(inputs=[input_seq, input_bg, input_dt], outputs=output, name="CombinedLSTMModel")
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
        if self.summary:
            print("Model input shapes:", input_seq.shape, input_bg.shape, input_dt.shape)
            model.summary()
        return model
        
    def train(self, X_train, X_bg_train, X_dt_train, y_train,
              X_val, X_bg_val, X_dt_val, y_val,
              epochs=10, batch_size=32, verbose='auto',
              checkpoint_path="best_model.h5",
              patience=10):

        # Make sure the parent folder exists
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        # Callbacks: save best to disk + restore best in memory during fit
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        earlystop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )

        # Train
        history = self.model.fit(
            [X_train, X_bg_train, X_dt_train], y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_val, X_bg_val, X_dt_val], y_val),
            verbose=verbose,
            callbacks=[checkpoint, earlystop]
        )

        # Reload the best checkpoint from disk to ensure perfect parity
        # (also restores optimizer state if you continue training later)

        self.model = load_model(checkpoint_path)
        print(f"[train] Reloaded best model from: {checkpoint_path}")

        return history

    
    def predict(self, X_test, X_bg_test, X_dt_test):
        y_pred = self.model.predict([X_test, X_bg_test, X_dt_test])
        return y_pred
    
    def save(self, file_path):
        self.model.save(file_path)
    
    def load(self, file_path):
        self.model = load_model(file_path)
    

def _extract_ids(ls):
    """Return 1-D array of IDs from ls' last-3rd element per row.
       Works for: list of arrays, 1-D object array, or 2-D array."""
    arr = np.asarray(ls, dtype=object)
    if arr.ndim == 1:
        return np.array([row[-3] for row in arr], dtype=object)
    elif arr.ndim == 2:
        return arr[:, -3]
    else:
        raise ValueError(f"Unsupported ls ndim={arr.ndim}")


def _split_ids(ls, test_size=0.2, val_size=None, random_state=None):
    ids = _extract_ids(ls)
    unique_ids = np.unique(ids)

    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )
    if val_size is None:
        return train_ids, None, test_ids

    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_size, random_state=random_state
    )
    return train_ids, val_ids, test_ids


def _select_by_ids(X, X_bg, dt, y, ls, seq_map, keep_ids):
    ids = _extract_ids(ls)
    mask = np.isin(ids, keep_ids)
    # ensure numpy for boolean indexing
    X      = np.asarray(X)
    X_bg   = np.asarray(X_bg)
    dt     = np.asarray(dt)
    y      = np.asarray(y)
    ls     = np.asarray(ls, dtype=object)
    seq_map= np.asarray(seq_map)
    return X[mask], X_bg[mask], dt[mask], y[mask], ls[mask], seq_map[mask]


def split_dataset(X_normalized, X_bg_normalized, dt, y, ls, seq_map,
                  test_size=0.2, val_size=0.2, random_state=None):
    train_ids, val_ids, test_ids = _split_ids(
        ls, test_size=test_size, val_size=val_size, random_state=random_state
    )

    X_train, X_bg_train, dt_train, y_train_, ls_train, seq_map_train = _select_by_ids(
        X_normalized, X_bg_normalized, dt, y, ls, seq_map, train_ids
    )
    X_val, X_bg_val, dt_val, y_val_, ls_val, seq_map_val = _select_by_ids(
        X_normalized, X_bg_normalized, dt, y, ls, seq_map, val_ids
    )
    X_test, X_bg_test, dt_test, y_test_, ls_test, seq_map_test = _select_by_ids(
        X_normalized, X_bg_normalized, dt, y, ls, seq_map, test_ids
    )

    # shapes for model
    dt_train = np.asarray(dt_train, dtype=np.float32).reshape(-1, 1, 1)
    dt_val   = np.asarray(dt_val,   dtype=np.float32).reshape(-1, 1, 1)
    dt_test  = np.asarray(dt_test,  dtype=np.float32).reshape(-1, 1, 1)

    y_train = np.asarray(y_train_).reshape(-1, 1)
    y_val   = np.asarray(y_val_).reshape(-1, 1)
    y_test  = np.asarray(y_test_).reshape(-1, 1)

    return (X_train, X_val, X_test,
            X_bg_train, X_bg_val, X_bg_test,
            dt_train, dt_val, dt_test,
            y_train, y_val, y_test,
            ls_train, ls_val, ls_test,
            seq_map_train, seq_map_val, seq_map_test)


def split_predictions_to_series(original_data, seq_map, y_values, seq_length):
    split_y = [[] for _ in range(len(original_data))]
    for idx, seq_idx in enumerate(seq_map):
        split_y[seq_idx].append(np.atleast_1d(y_values[idx]))  # ensures 1D array

    split_y = [np.concatenate(series) if series else np.array([]) for series in split_y]
    return split_y
