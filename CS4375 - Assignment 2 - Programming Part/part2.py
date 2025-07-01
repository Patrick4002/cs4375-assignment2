import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
import itertools
import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo

class NeuralNet:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = []
        self.histories = []
        
        # Define hyperparameter combinations to test
        self.hyperparameters = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (32, 32), (64, 64), (128, 64), (64, 32, 16)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
            'batch_size': [16, 32, 64, 'auto'],
            'max_iter': [200, 500, 1000]
        }
    
    def load_data(self):
        """Load the heart disease dataset from UCI ML repository"""
        print("Loading Heart Disease dataset from UCI ML repository...")
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print("\nDataset Info:")
        print(heart_disease.metadata)
        print("\nVariable Information:")
        print(heart_disease.variables)
        
        return X, y
    
    def preprocess(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
            
        X.fillna(X.mean(), inplace=True)

    # only fill y if there are actually missing values
        if  y.isnull().any():
            most_freq = y.mode().iloc[0]
            y.fillna(most_freq, inplace=True)

    # … the rest of your preprocessing …

        """
        Preprocess the data:
        - Handle null values
        - Ensure data integrity
        - Standardize attributes
        """
        print("\nPreprocessing data...")
        
        # Check for null values
        print(f"Null values in features: {X.isnull().sum().sum()}")
        print(f"Null values in target: {y.isnull().sum().sum()}")
        
        # Handle null values
        X = X.fillna(X.mean())  # Fill with mean for numerical
        y = y.fillna(y.mode()[0])  # Fill with mode for categorical
        
        # Ensure data integrity
        assert X.isnull().sum().sum() == 0, "Features still contain null values"
        assert y.isnull().sum().sum() == 0, "Target still contains null values"
        
        # Convert categorical variables to numerical if needed
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y.values.ravel())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_evaluate(self):
        """
        Train and evaluate neural networks with different hyperparameter combinations
        """
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        print(f"\nStarting hyperparameter optimization...")
        print(f"Input dimension: {self.X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(self.y_train))}")
        
        # Generate all combinations of hyperparameters
        keys = list(self.hyperparameters.keys())
        values = list(self.hyperparameters.values())
        combinations = list(itertools.product(*values))
        
        print(f"Total combinations to test: {len(combinations)}")
        
        # Limit combinations for practical runtime
        if len(combinations) > 30:
            np.random.seed(42)
            selected_indices = np.random.choice(len(combinations), 30, replace=False)
            combinations = [combinations[i] for i in selected_indices]
            print(f"Sampling {len(combinations)} combinations for practical runtime...")
        
        for i, combo in enumerate(combinations):
            print(f"\nTesting combination {i+1}/{len(combinations)}")
            
            # Create hyperparameter dictionary
            hyperparams = dict(zip(keys, combo))
            print(f"Hyperparameters: {hyperparams}")
            
            try:
                # Create and train model
                model = MLPClassifier(
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=10,
                    **hyperparams
                )
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(self.y_train, train_pred)
                test_accuracy = accuracy_score(self.y_test, test_pred)
                train_mse = mean_squared_error(self.y_train, train_pred)
                test_mse = mean_squared_error(self.y_test, test_pred)
                
                # Store results
                result = {
                    'model_id': i+1,
                    'hidden_layer_sizes': str(hyperparams['hidden_layer_sizes']),
                    'activation': hyperparams['activation'],
                    'solver': hyperparams['solver'],
                    'learning_rate_init': hyperparams['learning_rate_init'],
                    'alpha': hyperparams['alpha'],
                    'batch_size': hyperparams['batch_size'],
                    'max_iter': hyperparams['max_iter'],
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'n_iter': model.n_iter_,
                    'loss': model.loss_
                }
                
                self.results.append(result)
                
                # Store "history" (loss curve simulation)
                history = {
                    'model_id': i+1,
                    'loss_curve': model.loss_curve_,
                    'hyperparams': hyperparams
                }
                self.histories.append(history)
                
                print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
                print(f"Iterations: {model.n_iter_}, Final Loss: {model.loss_:.4f}")
                
            except Exception as e:
                print(f"Error with combination {i+1}: {str(e)}")
                continue
        
        print(f"\nCompleted testing {len(self.results)} models successfully.")
        return self.results, self.histories
    
    def plot_results(self):
        """Plot model history and results"""
        if not self.histories:
            print("No training histories to plot.")
            return
        
        # Plot training histories (loss curves)
        plt.figure(figsize=(20, 12))
        
        # Split into multiple subplots
        n_models = len(self.histories)
        models_per_plot = min(8, n_models)
        n_plots = (n_models + models_per_plot - 1) // models_per_plot
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * models_per_plot
            end_idx = min(start_idx + models_per_plot, n_models)
            
            plt.subplot(1, n_plots, plot_idx + 1)
            
            for i in range(start_idx, end_idx):
                history = self.histories[i]
                model_id = history['model_id']
                loss_curve = history['loss_curve']
                
                plt.plot(loss_curve, label=f'Model {model_id}', alpha=0.7)
            
            plt.title(f'Loss Curves (Models {start_idx+1}-{end_idx})')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot results summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            plt.figure(figsize=(15, 10))
            
            # Test accuracy vs solver
            plt.subplot(2, 2, 1)
            solver_acc = results_df.groupby('solver')['test_accuracy'].mean()
            solver_acc.plot(kind='bar')
            plt.title('Average Test Accuracy by Solver')
            plt.ylabel('Test Accuracy')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Learning rate vs accuracy
            plt.subplot(2, 2, 2)
            for lr in results_df['learning_rate_init'].unique():
                mask = results_df['learning_rate_init'] == lr
                plt.scatter(results_df[mask]['train_accuracy'], 
                           results_df[mask]['test_accuracy'], 
                           label=f'LR={lr}', alpha=0.7)
            plt.xlabel('Train Accuracy')
            plt.ylabel('Test Accuracy')
            plt.title('Train vs Test Accuracy by Learning Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Alpha (regularization) effect
            plt.subplot(2, 2, 3)
            alpha_acc = results_df.groupby('alpha')['test_accuracy'].mean()
            alpha_acc.plot(kind='bar')
            plt.title('Average Test Accuracy by Alpha (Regularization)')
            plt.ylabel('Test Accuracy')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Activation function comparison
            plt.subplot(2, 2, 4)
            activation_acc = results_df.groupby('activation')['test_accuracy'].mean()
            activation_acc.plot(kind='bar')
            plt.title('Average Test Accuracy by Activation Function')
            plt.ylabel('Test Accuracy')
            plt.xticks(rotation=0)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def print_results_table(self):
        """Print results in a formatted table"""
        if not self.results:
            print("No results to display.")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Sort by test accuracy
        results_df = results_df.sort_values('test_accuracy', ascending=False)
        
        print("\n" + "="*150)
        print("NEURAL NETWORK HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*150)
        
        # Display top 10 models
        print("\nTOP 10 MODELS BY TEST ACCURACY:")
        print("-"*150)
        
        columns_to_show = [
            'model_id', 'hidden_layer_sizes', 'activation', 'solver', 
            'learning_rate_init', 'alpha', 'batch_size', 'max_iter',
            'train_accuracy', 'test_accuracy', 'train_mse', 'test_mse', 'n_iter'
        ]
        
        top_models = results_df.head(10)[columns_to_show]
        
        # Format the display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print(top_models.to_string(index=False))
        
        # Summary statistics
        print(f"\n\nSUMMARY STATISTICS:")
        print("-"*50)
        print(f"Total models trained: {len(results_df)}")
        print(f"Best test accuracy: {results_df['test_accuracy'].max():.4f}")
        print(f"Average test accuracy: {results_df['test_accuracy'].mean():.4f}")
        print(f"Standard deviation: {results_df['test_accuracy'].std():.4f}")
        
        # Best hyperparameters
        best_model = results_df.iloc[0]
        print(f"\nBEST MODEL HYPERPARAMETERS:")
        print("-"*50)
        for param in ['hidden_layer_sizes', 'activation', 'solver', 
                     'learning_rate_init', 'alpha', 'batch_size']:
            print(f"{param}: {best_model[param]}")
        
        return results_df

def main():
    """Main function to run the neural network optimization"""
    print("Neural Network Hyperparameter Optimization with Scikit-Learn")
    print("="*60)
    
    # Initialize the neural network class
    nn = NeuralNet()
    
    try:
        # Load data
        X, y = nn.load_data()
        
        # Preprocess data
        nn.preprocess(X, y)
        
        # Train and evaluate models
        results, histories = nn.train_evaluate()
        
        # Display results
        results_df = nn.print_results_table()
        
        # Plot results
        nn.plot_results()
        
        print("\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()