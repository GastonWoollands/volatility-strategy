import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#------------------------------------------------------------------------------------------------

def calculate_volatility(df, window=30):
    df['volatility'] = df['log_return'].rolling(window=window).std() * np.sqrt(252)
    df.dropna(inplace=True)
    
    return df

#------------------------------------------------------------------------------------------------

class DataPreprocessor:
    def __init__(self, seq_length: int, batch_size: int, scaler = None, extra_vars:list = None, output_size: int = 1):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.scaler = scaler if scaler else StandardScaler()
        self.extra_vars = extra_vars if extra_vars else []
        self.output_size = output_size

    def create_sequences(self, df):
        """
        Crea secuencias de entrada (X) y etiquetas de salida (y) para el modelo.

        Args:
            df: DataFrame con las columnas necesarias para crear secuencias.

        Returns:
            X: Secuencias de entrada.
            y: Etiquetas de salida con múltiples días.
        """
        X, y = [], []
        for i in range(len(df) - self.seq_length - self.output_size + 1):
            sequence = df['log_return'].iloc[i:i + self.seq_length].values.reshape(-1, 1)

            if self.extra_vars:
                extra_data = df[self.extra_vars].iloc[i:i + self.seq_length].values
                sequence = np.concatenate([sequence, extra_data], axis=1)

            X.append(sequence)

            target = df['log_return'].iloc[i + self.seq_length:i + self.seq_length + self.output_size].values
            y.append(target)

        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def preprocess_data(self, df, predict=False):
        """
        Preprocess data and save scaler if a path is provided.
        """
        _df = df.copy()

        if not predict:
            _df.loc[:, 'log_return'] = self.scaler.fit_transform(df[['log_return']]).astype(np.float64)
        else:
            transformed_data = self.scaler.transform(df[['log_return']])
            _df.loc[:, 'log_return'] = transformed_data.flatten().astype(np.float64)
               
        X, y = self.create_sequences(_df)

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
        Make predictions for N periods ahead, supporting models with output_size > 1.

        - df: DataFrame with the input data for predictions.
        - n_days: Number of days/periods to predict.
        - seq_length: Length of the input sequence.
        - return: Array with the inverse-scaled predictions.
        """

        _df = df.iloc[-(seq_length + self.data_preprocessor.output_size):].copy()

        input_seq = self.data_preprocessor.preprocess_data(_df, predict=True).to(self.device)

        predictions = []

        if self.data_preprocessor.output_size == 1:
            with torch.no_grad():
                output, _ = self.model(input_seq)
                predictions.append(output.item())
        else:
            if self.data_preprocessor.output_size >= n_days:
                with torch.no_grad():
                    output, _ = self.model(input_seq)
                    predictions.append(output.squeeze().cpu().numpy())
            else:
                raise ValueError(f"Output size {self.data_preprocessor.output_size} is less than the number of days to predict {n_days}")

        predictions = np.concatenate(predictions, axis=0)[:n_days]
        if self.data_preprocessor.output_size == 1:
            pred_gru = self.data_preprocessor.scaler.inverse_transform(predictions.reshape(-1, 1))
        else:
            pred_gru = self.data_preprocessor.scaler.inverse_transform(predictions.reshape(-1, self.data_preprocessor.output_size))

        return pred_gru.flatten()
    
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
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device: str, epochs: int=1000, early_stopping_patience: int=20, seed: int=None, debug:bool=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.debug = debug
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
            self.model.train()
            running_train_loss = 0

            for input_seq, target in self.train_loader:
                input_seq, target = input_seq.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(input_seq, hidden_state=None)
                loss = self.criterion(output, target)
                loss.backward()
                if self.debug:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f'{name} grad: {param.grad}')
                self.optimizer.step()
                running_train_loss += loss.item()
            
            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            val_loss = self.validate()
            val_losses.append(val_loss)
            
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
        self.model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for input_seq, target in self.test_loader:
                input_seq, target = input_seq.to(self.device), target.to(self.device)
                output, _ = self.model(input_seq, hidden_state=None)
                val_loss = self.criterion(output, target)
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

#------------------------------------------------------------------------------------------------

def predict_and_evaluate(model, X_test, y_test, device, output_size, scaler, df):
    """
    Predict and evaluate the model.
    """
    predictions = []
    true_values = []
    hidden_state = None

    with torch.no_grad():
        for i in range(len(X_test)):
            input_seq = X_test[i].unsqueeze(0).to(device)
            target = y_test[i].to(device)

            if output_size > 1:  
                output, hidden_state = model(input_seq, hidden_state)
                predictions.append(output.squeeze(0).cpu().numpy())
                true_values.append(target.cpu().numpy())
            else:
                output, hidden_state = model(input_seq, hidden_state)
                predictions.append(output.item())
                true_values.append(target.item())

                output_expanded = output.view(1, 1, -1).expand(1, 1, input_seq.size(-1))
                input_seq = torch.cat((input_seq[:, 1:, :], output_expanded), dim=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(-1, output_size)
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1)).reshape(-1, output_size)


    mse = mean_squared_error(true_values.flatten(), predictions.flatten())
    mae = mean_absolute_error(true_values.flatten(), predictions.flatten())
    r2 = r2_score(true_values.flatten(), predictions.flatten())

    mse_per_prediction = [mean_squared_error(true_values[i].flatten(), predictions[i].flatten()) for i in range(len(predictions))]
    mae_per_prediction = [mean_absolute_error(true_values[i].flatten(), predictions[i].flatten()) for i in range(len(predictions))]

    return mse, mae, r2, mse_per_prediction, mae_per_prediction, true_values, predictions

#------------------------------------------------------------------------------------------------

class MultiModelPredictor:
    def __init__(self, models: dict, data_preprocessor: DataPreprocessor, device: str = 'cpu'):
        """
        Predictor for multiple models with different horizons.

        - models: Dictionary with models and their prediction horizons. 
                  Example: {5: model_5, 10: model_10, 20: model_20, 30: model_30}
        - data_preprocessor: Instance of DataPreprocessor.
        - device: Device where the models will be executed ('cpu' or 'cuda').
        """
        self.models = {int(k): v.to(device) for k, v in models.items()}
        self.data_preprocessor = data_preprocessor
        self.device = device

    def predict(self, df, n_days: int, seq_length: int):
        """
        Make predictions for n days using the appropriate model.

        - df: DataFrame with the input data.
        - n_days: Number of days to predict.
        - seq_length: Length of the input sequence.
        - return: Array with the inverse-scaled predictions.
        """
        if n_days not in self.models:
            raise ValueError(f"No model available for {n_days} days. Available models: {list(self.models.keys())}")

        model = self.models[n_days]
        _df = df.iloc[-(seq_length + self.data_preprocessor.output_size):].copy()

        input_seq = self.data_preprocessor.preprocess_data(_df, predict=True).to(self.device)

        with torch.no_grad():
            output, _ = model(input_seq)
            predictions = output.squeeze().cpu().numpy()

        if predictions.size % n_days != 0:
            raise ValueError(f"Cannot reshape predictions of size {predictions.size} into shape (-1, {n_days})")

        predictions = self.data_preprocessor.scaler.inverse_transform(predictions.reshape(-1, n_days)).flatten()
        return predictions

    def predict_all(self, df, seq_length: int):
        """Predictions for all horizons"""
        results = {}
        for n_days in sorted(self.models.keys()):
            try:
                results[n_days] = self.predict(df, n_days, seq_length)
            except ValueError as e:
                print(f"Skipping predictions for {n_days} days due to error: {e}")
        return results