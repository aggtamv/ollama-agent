import os
import sys
import requests
import datetime
import pandas as pd
from io import BytesIO
from io import StringIO
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
import traceback
import contextlib

load_dotenv()

@tool
def read_csv(filepath: str) -> str:
    """Reads a CSV file and returns information about its structure and first few rows."""
    try:
        df = pd.read_csv(filepath)
        info = f"CSV loaded successfully!\n"
        info += f"Shape: {df.shape}\n"
        info += f"Columns: {list(df.columns)}\n\n"
        info += "First 5 rows:\n"
        info += df.head().to_string()
        return info
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

@tool
def write_output(filepath: str, contents: str) -> str:
    """Writes text to a file."""
    try:
        with open(filepath, "w") as f:
            f.write(contents)
        return f"Successfully written to {filepath}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def create_classifier(filepath: str) -> str:
    """
    Trains an SVM classifier to predict NBA player positions from stats.
    Outputs predictions to sol.csv with columns: player, actual position, predicted position.
    """
    try:
        print(f"[DEBUG] Called create_classifier with: {filepath}")
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        import numpy as np

        # Verify file exists
        if not os.path.exists(filepath):
            return f"Error: File {filepath} not found."

        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Clean the data
        df = df[df["Player"] != "League Average"]
        df = df.drop(columns=['Rk', 'Team', 'Awards'], errors='ignore')
        df = df.drop_duplicates(subset=['Player'], keep='last')
        
        if 'Player' not in df.columns or 'Pos' not in df.columns:
            return f"Error: CSV must contain 'Player' and 'Pos' columns. Found columns: {list(df.columns)}"

        # Extract player names and target labels
        player_names = df['Player'].reset_index(drop=True)
        y = df['Pos'].reset_index(drop=True)

        # Drop non-numeric/statistical columns before modeling
        X = df.drop(columns=['Player', 'Pos'], errors='ignore')

        # Ensure only numeric columns are kept
        X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        
        if X.empty:
            return f"Error: No numeric columns found for training. Available columns: {list(df.columns)}"

        # Fill missing values with mean
        X = X.fillna(X.mean(numeric_only=True))
        
        # Handle any remaining NaN values
        X = X.fillna(0)

        # Verify we have enough data
        if len(X) < 10:
            return f"Error: Not enough data for training. Only {len(X)} samples found."

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split dataset into train/test (80% / 20%)
        split_index = max(1, int(0.8 * len(X_scaled)))  # Ensure at least 1 training sample
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        player_test = player_names[split_index:]

        if len(X_test) == 0:
            return "Error: No test data available after split. Dataset too small."

        # Train SVM with better parameters for multiclass classification
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Create result DataFrame with exact format requested
        results = pd.DataFrame({
            'Player name': player_test.values,
            "player's actual position": y_test.values,
            'predicted position': y_pred
        })

        # Save predictions to sol.csv with absolute path for clarity
        output_path = os.path.abspath('sol.csv')
        results.to_csv(output_path, index=False)
        
        # Verify file was created
        if not os.path.exists(output_path):
            return f"Error: Failed to create output file at {output_path}"
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        
        result_summary = f"""✅ SVM Classification Complete!
        
Data Summary:
- Total samples: {len(df)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features used: {X.shape[1]}
- Feature columns: {list(X.columns)}

Results:
- Accuracy: {accuracy:.3f}
- Output file: {output_path}
- File size: {os.path.getsize(output_path)} bytes
- Predictions count: {len(results)}

Classification Report:
{report}

✅ SUCCESS: sol.csv has been created with {len(results)} predictions!
File location: {output_path}
"""
        
        return result_summary

    except Exception as e:
        error_details = f"Error in create_classifier: {str(e)}\n"
        error_details += f"Traceback:\n{traceback.format_exc()}"
        return error_details

@tool
def execute_python(code: str, context_vars: Dict[str, Any] = None) -> str:
    """
    Executes Python code and returns the output.
    
    Args:
        code: Python code to execute
        context_vars: Optional dictionary of variables to make available in execution context
    
    Returns:
        String containing the output, error messages, or results
    """
    if context_vars is None:
        context_vars = {}
    
    # Create a safe execution environment
    safe_globals = {
        '__builtins__': {
            '__import__': __import__,
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
        },
        'pd': pd,
        'os': os,
        'datetime': datetime,
        'create_classifier': create_classifier
    }
    
    # Add context variables
    safe_globals.update(context_vars)
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Execute the code
        exec(code, safe_globals)
        output = captured_output.getvalue()
        
        # If there's no printed output, try to get the last expression result
        if not output.strip():
            try:
                result = eval(code, safe_globals)
                if result is not None:
                    output = str(result)
            except:
                pass
        
        return output if output.strip() else "Code executed successfully (no output)"
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}\n"
        error_msg += traceback.format_exc()
        return error_msg
    
    finally:
        sys.stdout = old_stdout