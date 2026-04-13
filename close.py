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
from statsmodels.tsa.arima.model import ARIMA


# ================= 配置类 =================
class Config:
    FILE_MAIN = '日度数据.csv'
    FILE_TECH = '技术指标.csv'

    TARGET_COL = 'close'
    OUTPUT_EXCEL = 'prediction_results_all_windows.xlsx'

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

    HIDDEN_DIM = 64
    LAYERS = 2
    DROPOUT = 0.1

    # ARIMA 参数
    ARIMA_ORDER = (1, 0, 1)

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
        return None, None

    if df_main.empty or df_tech.empty:
        print("错误：读取到的数据为空！")
        return None, None

    for df in [df_main, df_tech]:
        if 'trade_date' not in df.columns:
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                df.rename(columns={date_cols[0]: 'trade_date'}, inplace=True)
            else:
                print("错误：未找到日期列")
                return None, None

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
        return None, None

    if Config.TARGET_COL not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df.rename(columns={close_candidates[0]: Config.TARGET_COL}, inplace=True)
        else:
            print(f"错误：未找到目标列 '{Config.TARGET_COL}'")
            return None, None

    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in all_numeric if c != Config.TARGET_COL and 'Unnamed' not in c]

    df[features] = df[features].ffill().bfill().fillna(0)
    df[Config.TARGET_COL] = df[Config.TARGET_COL].ffill().bfill().fillna(0)

    final_df = df[['trade_date', Config.TARGET_COL] + features].copy().sort_values('trade_date').reset_index(drop=True)

    print(f"数据加载完成。总行数: {len(final_df)}, 特征数: {len(features)}")
    return final_df, features


# ================= 模型定义 =================
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, Config.HIDDEN_DIM, Config.LAYERS, batch_first=True, dropout=Config.DROPOUT)
        self.fc = nn.Linear(Config.HIDDEN_DIM, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, Config.HIDDEN_DIM, Config.LAYERS, batch_first=True,
                            bidirectional=True, dropout=Config.DROPOUT)
        self.fc = nn.Linear(Config.HIDDEN_DIM * 2, 1)
        self.attn = nn.Linear(Config.HIDDEN_DIM * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


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


# ================= 深度学习训练函数 =================
def train_model(model_class, df, features, window):
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
        return 0.0, 0.0, [], []

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = model_class(len(features)).to(Config.DEVICE)
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
            print(f"{model_class.__name__} | Epoch {epoch + 1}/{Config.EPOCHS}, "
                  f"Train Loss: {epoch_loss / len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")

        if early_stopping.early_stop:
            print(f"{model_class.__name__} 早停于 Epoch {epoch + 1}")
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
        return 0.0, 0.0, [], []

    final_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    final_act = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    try:
        mape = mean_absolute_percentage_error(final_act, final_pred) * 100
        rmse = np.sqrt(mean_squared_error(final_act, final_pred))
    except Exception:
        mape, rmse = 0.0, 0.0

    return mape, rmse, final_pred, final_act


# ================= ARIMA 滚动预测 =================
def run_arima_rolling(df, window):
    print(f"正在运行 ARIMA{Config.ARIMA_ORDER} 滚动预测 (Window={window})...")

    data = df[Config.TARGET_COL].values
    total_len = len(data)
    train_size = int(total_len * 0.8)

    train_data = list(data[:train_size])
    test_data = list(data[train_size:])

    if len(test_data) <= window:
        return 0.0, 0.0, [], []

    history = train_data.copy()
    predictions = []
    actuals = []

    for t in range(len(test_data)):
        try:
            model = ARIMA(history, order=Config.ARIMA_ORDER)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)[0]
        except Exception as e:
            print(f"ARIMA step {t} error: {e}")
            yhat = history[-1] if len(history) > 0 else 0.0

        predictions.append(yhat)
        actuals.append(test_data[t])

        # 真正滚动：加入真实值
        history.append(test_data[t])

    predictions = np.array(predictions[window:])
    actuals = np.array(actuals[window:])

    if len(predictions) == 0:
        return 0.0, 0.0, [], []

    try:
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
    except Exception:
        mape, rmse = 0.0, 0.0

    return mape, rmse, predictions, actuals


# ================= 保存预测结果到 Excel =================
def save_predictions_to_excel(all_results, filename):
    print(f"\n正在保存预测结果到 Excel: {filename}")
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for w_size, df_res in all_results.items():
                sheet_name = f"Window_{w_size}"
                cols = ['trade_date', 'Actual', 'ARIMA', 'LSTM', 'BiLSTM-Attention', 'Transformer']
                existing_cols = [c for c in cols if c in df_res.columns]
                df_res[existing_cols].to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  - Sheet '{sheet_name}' 已保存，行数: {len(df_res)}")
        print("Excel 导出完成。")
    except Exception as e:
        print(f"保存 Excel 失败: {e}")
        print("请确认已安装 openpyxl：pip install openpyxl")


# ================= 主程序 =================
if __name__ == "__main__":
    df, features = load_data()

    if df is None or len(df) == 0:
        print("程序终止：无法加载有效数据。")
        sys.exit(1)

    total_len = len(df)
    if total_len < 50:
        print("错误：数据量太小。")
        sys.exit(1)

    train_len_rough = int(total_len * 0.8)
    test_len_rough = total_len - train_len_rough

    print(f"Train size: {train_len_rough}, Test size: {test_len_rough}")
    print(f"ARIMA Order: {Config.ARIMA_ORDER}")

    # 指标结果表（仅打印，不保存 CSV）
    results_table = {'Model': ['ARIMA', 'LSTM', 'BiLSTM-Attention', 'Transformer']}

    # Excel 预测结果
    excel_data_store = {}

    for w in Config.WINDOW_SIZES:
        print(f"\n================ Window = {w} ================")

        dl_test_len = test_len_rough - w
        if dl_test_len <= 0:
            print(f"跳过 Window={w}: 窗口过大。")
            continue

        results_table[f'Window={w} (MAPE)'] = []
        results_table[f'Window={w} (RMSE)'] = []

        actual_values_full = df[Config.TARGET_COL].values
        dates_full = df['trade_date'].values

        start_idx = train_len_rough + w
        end_idx = train_len_rough + w + dl_test_len

        current_dates = dates_full[start_idx:end_idx]
        actual_values = actual_values_full[start_idx:end_idx]

        df_window_res = pd.DataFrame({
            'trade_date': current_dates,
            'Actual': actual_values
        })

        # ===== ARIMA =====
        arima_mape, arima_rmse, arima_pred, arima_act = run_arima_rolling(df, w)

        if len(arima_pred) == 0:
            results_table[f'Window={w} (MAPE)'].append("N/A")
            results_table[f'Window={w} (RMSE)'].append("N/A")
            df_window_res['ARIMA'] = np.nan
        else:
            results_table[f'Window={w} (MAPE)'].append(f"{arima_mape:.4f}")
            results_table[f'Window={w} (RMSE)'].append(f"{arima_rmse:.4f}")
            df_window_res['ARIMA'] = arima_pred[:len(df_window_res)]

        # ===== 深度学习模型 =====
        models = [
            ('LSTM', LSTMModel),
            ('BiLSTM-Attention', BiLSTMAttention),
            ('Transformer', TransformerModel)
        ]

        for name, cls in models:
            print(f"Training {name}...")
            mape, rmse, pred, act = train_model(cls, df, features, w)

            if len(pred) == 0:
                results_table[f'Window={w} (MAPE)'].append("N/A")
                results_table[f'Window={w} (RMSE)'].append("N/A")
                df_window_res[name] = np.nan
            else:
                results_table[f'Window={w} (MAPE)'].append(f"{mape:.4f}")
                results_table[f'Window={w} (RMSE)'].append(f"{rmse:.4f}")
                df_window_res[name] = pred[:len(df_window_res)]

        excel_data_store[w] = df_window_res

    final_df_res = pd.DataFrame(results_table)
    cols = ['Model'] + [c for c in final_df_res.columns if c != 'Model']
    final_df_res = final_df_res[cols]

    print("\n================ 最终评价指标表 ================")
    print(final_df_res)

    # ===== 输出预测结果 Excel =====
    if excel_data_store:
        save_predictions_to_excel(excel_data_store, Config.OUTPUT_EXCEL)
    else:
        print("没有生成预测结果，跳过 Excel 导出。")

    print("\n所有任务完成。")