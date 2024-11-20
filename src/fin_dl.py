import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


#------------------------------------------------------------------------------------------------

def calculate_volatility(df, window=30):
    df['volatility'] = df['log_return'].rolling(window=window).std() * np.sqrt(252)
    df.dropna(inplace=True)
    
    return df

#------------------------------------------------------------------------------------------------

def create_sequences(df, seq_length):
    X = []
    y = []
    
    for i in range(len(df) - seq_length):
        X.append(df['log_return'].iloc[i:i+seq_length].values)
        y.append(df['log_return'].iloc[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

#------------------------------------------------------------------------------------------------

class DataPreprocessor:
    def __init__(self, seq_length: int, batch_size: int, scaler = None, extra_vars:list = None):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.scaler = scaler if scaler else StandardScaler()
        self.extra_vars = extra_vars if extra_vars else []

    def create_sequences(self, df):
        """
        Create sequences of logarithmic returns and their associated volatility,
        including additional variables if specified.
        """
        X, y = [], []
        for i in range(len(df) - self.seq_length):
            sequence = df['log_return'].iloc[i:i + self.seq_length].values.reshape(-1, 1)
            
            if self.extra_vars:
                extra_data = df[self.extra_vars].iloc[i:i + self.seq_length].values
                sequence = np.concatenate([sequence, extra_data], axis=1)

            X.append(sequence)
            y.append(df['log_return'].iloc[i + self.seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def preprocess_data(self, df, predict=False):
        """
        Preprocess data and save scaler if a path is provided.
        """
        if not predict:
            df.loc[:, 'log_return'] = self.scaler.fit_transform(df[['log_return']]).astype(np.float64)
        else:
            transformed_data = self.scaler.transform(df[['log_return']])
            df.loc[:, 'log_return'] = transformed_data.flatten().astype(np.float64)
               
        X, y = self.create_sequences(df)

        if predict:
            input_seq = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0)
            return input_seq
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)
        
        train_data = TensorDataset(X_train, y_train)
        test_data  = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, X_train, X_test, y_train, y_test

    @property
    def scaler(self):
        return self._scaler
    
    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

#------------------------------------------------------------------------------------------------

class Predictor:
    def __init__(self, model, data_preprocessor, device='cpu'):
        """
        Init predictor.
        
        - model: Model to predict.
        - data_preprocessor: Instance of DataPreprocessor.
        - device: Device where the model will be executed ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.data_preprocessor = data_preprocessor
        self.device = device

    def predict(self, df, n_days: int, seq_length: int):
        """
        Make predictions for N periods ahead.

        - df: DataFrame with the input data for predictions.
        - n_days: Number of days/periods to predict.
        - seq_length: Length of the input sequence.
        - return: List of predictions.
        """
        input_seq = self.data_preprocessor.preprocess_data(df.iloc[-seq_length - 1:], predict=True).to(self.device)

        predictions = []
        hidden_state = None
        current_input = input_seq

        for _ in range(n_days):
            with torch.no_grad():
                output, hidden_state = self.model(current_input, hidden_state)

            predicted_log_return = output.item()

            predicted_value = self.data_preprocessor.scaler.inverse_transform([[predicted_log_return]])[0][0]

            predictions.append(predicted_value)

            current_input = torch.tensor(np.roll(current_input.cpu().numpy(), -1, axis=1), dtype=torch.float32).to(self.device)
            current_input[0, -1, 0] = predicted_log_return

        return predictions

#------------------------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):        
        if hidden_state is None:
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        else:
            h0, c0 = hidden_state
        
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        out = self.fc(lstm_out[:, -1, :])
        
        return out, (h_n, c_n)
    
#------------------------------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, device='cpu'):
        """
        Args:
            input_size: Input tensor dimension.
            hidden_size: Hidden state dimension.
            output_size: Output tensor dimension.
            dropout: Dropout probability (0.0 if not desired).
            device: Device where the model will run ('cpu' or 'cuda').
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(self.device)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor.
            hidden_state: Hidden state initial (optional).

        Returns:
            out: Output tensor.
            hidden_state: Last hidden state of the GRU.
        """
        if hidden_state is None:
            hidden_state = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        x = x.to(self.device)
        out, hidden_state = self.gru(x, hidden_state)
        if self.dropout:
            out = self.dropout(out)

        # out = self.fc(out)
        out = self.fc(out[:, -1, :])
        return out, hidden_state

#------------------------------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, epochs=1000, early_stopping_patience=20, seed=None):
        self._model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            self._model.train()
            running_train_loss = 0
            
            for input_seq, target in self.train_loader:
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                
                hidden_state = None
                output, hidden_state = self._model(input_seq, hidden_state)

                loss = self.criterion(output.squeeze(-1), target)
                loss.backward()
                self.optimizer.step()
                
                running_train_loss += loss.item()
            
            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            
            # Validación
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        return train_losses, val_losses
    
    def validate(self):
        self._model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for input_seq, target in self.test_loader:
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)
                
                output, _ = self._model(input_seq, hidden_state=None)
                val_loss = self.criterion(output.squeeze(-1), target)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(self.test_loader)
        return avg_val_loss

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def train_loader(self):
        return self._train_loader

    @train_loader.setter
    def train_loader(self, value):
        self._train_loader = value

    @property
    def test_loader(self):
        return self._test_loader

    @test_loader.setter
    def test_loader(self, value):
        self._test_loader = value

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, value):
        self._criterion = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @property
    def early_stopping_patience(self):
        return self._early_stopping_patience

    @early_stopping_patience.setter
    def early_stopping_patience(self, value):
        self._early_stopping_patience = value