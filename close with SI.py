import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import math
import sys
import copy
import random


# ================= 配置类 =================
class Config:
    FILE_MAIN = '日度数据_SI.csv'
    FILE_TECH = '技术指标.csv'

    TARGET_COL = 'close'
    WINDOW_SIZES = [1, 5, 15, 30]

    EPOCHS = 120
    BATCH_SIZE = 32
    LR = 0.001

    # 早停配置
    EARLY_STOP_PATIENCE = 15
    EARLY_STOP_MIN_DELTA = 0.0001

    # Transformer 配置
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_N_HEAD = 8
    TRANSFORMER_ENC_LAYERS = 2
    TRANSFORMER_DEC_LAYERS = 2
    TRANSFORMER_DIM_FF = 256
    TRANSFORMER_DROPOUT = 0.1

    # 指定情绪指标列
    SENTIMENT_COL = 'SI'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 初始化设置 =================
warnings.filterwarnings('ignore')
print(f"Running on: {Config.DEVICE}")


# ================= 早停机制 =================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_wts = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)


# ================= 数据加载 =================
def load_data():
    print("正在加载 CSV 数据...")

    def read_csv_safe(filepath):
        try:
            return pd.read_csv(filepath)
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
            return None

    df_main = read_csv_safe(Config.FILE_MAIN)
    df_tech = read_csv_safe(Config.FILE_TECH)

    if df_main is None or df_tech is None:
        return None, None, None

    if df_main.empty or df_tech.empty:
        print("错误：读取到的数据为空！")
        return None, None, None

    for df in [df_main, df_tech]:
        if 'trade_date' not in df.columns:
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                df.rename(columns={date_cols[0]: 'trade_date'}, inplace=True)
            else:
                print("错误：未找到日期列")
                return None, None, None

        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')

    df_main = df_main.dropna(subset=['trade_date'])
    df_tech = df_tech.dropna(subset=['trade_date'])

    merge_keys = ['trade_date']
    if 'ts_code' in df_main.columns and 'ts_code' in df_tech.columns:
        merge_keys.append('ts_code')
    elif 'instrument' in df_main.columns and 'instrument' in df_tech.columns:
        merge_keys.append('instrument')

    df = pd.merge(df_main, df_tech, on=merge_keys, how='inner')

    if df.empty:
        print("错误：合并后数据为空！")
        return None, None, None

    if Config.TARGET_COL not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df.rename(columns={close_candidates[0]: Config.TARGET_COL}, inplace=True)
        else:
            print(f"错误：未找到目标列 '{Config.TARGET_COL}'")
            return None, None, None

    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in all_numeric if c != Config.TARGET_COL and 'Unnamed' not in c]

    if Config.SENTIMENT_COL not in features:
        print(f"错误：未找到情绪指标列 '{Config.SENTIMENT_COL}'")
        print(f"当前可用数值列为：{features}")
        return None, None, None

    # 基础特征（不含情绪指标）
    base_features = [c for c in features if c != Config.SENTIMENT_COL]

    # 融合特征（含情绪指标）
    fusion_features = base_features + [Config.SENTIMENT_COL]

    df[features] = df[features].ffill().bfill().fillna(0)
    df[Config.TARGET_COL] = df[Config.TARGET_COL].ffill().bfill().fillna(0)

    final_df = df[['trade_date', Config.TARGET_COL] + features].copy().sort_values('trade_date').reset_index(drop=True)

    print(f"数据加载完成。总行数: {len(final_df)}")

    return final_df, base_features, fusion_features


# ================= Transformer 模型定义 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            max_len = x.size(1)
            pe = torch.zeros(max_len, self.pe.size(2), device=x.device)
            position = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, device=x.device).float() *
                                 (-math.log(10000.0) / self.pe.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return x


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim):
        super(EncoderDecoderTransformer, self).__init__()

        d_model = Config.TRANSFORMER_D_MODEL
        nhead = Config.TRANSFORMER_N_HEAD
        num_encoder_layers = Config.TRANSFORMER_ENC_LAYERS
        num_decoder_layers = Config.TRANSFORMER_DEC_LAYERS
        dim_feedforward = Config.TRANSFORMER_DIM_FF
        dropout = Config.TRANSFORMER_DROPOUT

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        batch_size, seq_len, _ = src.shape

        x_enc = self.input_proj(src) * math.sqrt(self.d_model)
        x_enc = self.pos_encoder(x_enc)
        memory = self.transformer_encoder(x_enc)

        tgt = torch.zeros(batch_size, seq_len, self.d_model).to(src.device)
        x_dec = self.pos_decoder(tgt)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()

        output = self.transformer_decoder(
            tgt=x_dec, memory=memory, tgt_mask=causal_mask
        )

        last_output = output[:, -1, :]
        return self.fc_out(last_output)


class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.model = EncoderDecoderTransformer(input_dim)

    def forward(self, x):
        return self.model(x)


# ================= 数据集类 =================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return max(0, len(self.X) - self.window)

    def __getitem__(self, i):
        return self.X[i:i + self.window], self.y[i + self.window]


# ================= Transformer 训练函数 =================
def train_transformer(df, features, window):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data_X = scaler_X.fit_transform(df[features])
    data_y = scaler_y.fit_transform(df[[Config.TARGET_COL]])

    train_size = int(len(data_X) * 0.8)
    train_X, test_X = data_X[:train_size], data_X[train_size:]
    train_y, test_y = data_y[:train_size], data_y[train_size:]

    train_ds = TimeSeriesDataset(train_X, train_y, window)
    test_ds = TimeSeriesDataset(test_X, test_y, window)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return np.nan, np.nan

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = TransformerModel(len(features)).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOP_PATIENCE,
        min_delta=Config.EARLY_STOP_MIN_DELTA
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred.squeeze(), y_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                pred = model(X_batch)
                loss = criterion(pred.squeeze(), y_batch.squeeze())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}, "
                  f"Train Loss: {epoch_loss / len(train_loader):.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        if early_stopping.early_stop:
            print(f"Transformer 早停于 Epoch {epoch + 1}")
            early_stopping.load_checkpoint(model)
            break

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            p = model(X_batch).cpu().numpy().flatten()
            preds.extend(p)
            actuals.extend(y_batch.numpy().flatten())

    if len(preds) == 0:
        return np.nan, np.nan

    final_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    final_act = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    try:
        mape = mean_absolute_percentage_error(final_act, final_pred) * 100
        rmse = np.sqrt(mean_squared_error(final_act, final_pred))
    except Exception:
        mape, rmse = np.nan, np.nan

    return mape, rmse


# ================= 主程序 =================
if __name__ == "__main__":
    df, base_features, fusion_features = load_data()

    if df is None or len(df) == 0:
        print("程序终止：无法加载有效数据。")
        sys.exit(1)

    total_len = len(df)
    if total_len < 50:
        print("错误：数据量太小。")
        sys.exit(1)

    # 结果字典
    without_si = {}
    with_si = {}

    for w in Config.WINDOW_SIZES:
        print(f"\n================ Window = {w} ================")

        # 无情绪指标
        print("训练 Transformer（无情绪指标）...")
        mape_base, rmse_base = train_transformer(df, base_features, w)
        without_si[w] = (mape_base, rmse_base)

        # 有情绪指标
        print("训练 Transformer（有情绪指标：SI）...")
        mape_fusion, rmse_fusion = train_transformer(df, fusion_features, w)
        with_si[w] = (mape_fusion, rmse_fusion)

    result_table = pd.DataFrame({
        '模型': ['无情绪指标', '有情绪指标']
    })

    for w in Config.WINDOW_SIZES:
        result_table[f'Window={w}_MAPE'] = [
            round(without_si[w][0], 4) if not pd.isna(without_si[w][0]) else np.nan,
            round(with_si[w][0], 4) if not pd.isna(with_si[w][0]) else np.nan
        ]
        result_table[f'Window={w}_RMSE'] = [
            round(without_si[w][1], 4) if not pd.isna(without_si[w][1]) else np.nan,
            round(with_si[w][1], 4) if not pd.isna(with_si[w][1]) else np.nan
        ]

    print("\n================ 最终结果================\n")
    print(result_table.to_string(index=False))

    print("\n实验完成。")