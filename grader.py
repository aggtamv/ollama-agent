import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel


class GradingResult(BaseModel):
    score: float
    subscores: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    feedback: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def grade(transcript: str) -> GradingResult:
    """
    Grade the agent's performance on the NBA player's position classification task.
    
    The grading is based on:
    1. Whether the predictions file exists and is properly formatted
    2. Classification accuracy (1.0 - classification error)
    3. Proper train/test split (80/20)
    """
    
    # Initialize scoring components
    subscores = {}
    details = {}
    feedback_parts = []
    
    # Check if predictions file exists
    predictions_path = 'sol.csv'
    if not os.path.exists(predictions_path):
        return GradingResult(
            score=0.0,
            feedback="sol.csv file not found. Please save your predictions to sol.csv",
            details={"error": "missing_predictions_file"}
        )
    
    try:
        # Load predictions
        predictions_df = pd.read_csv(predictions_path)
        
        # Check if required columns exist
        actual_col = None
        predicted_col = None

        for col in predictions_df.columns:
            col_lower = col.lower()
            if 'actual' in col_lower:
                actual_col = col
            elif 'predicted' in col_lower:
                predicted_col = col

        if actual_col is None or predicted_col is None:
            return GradingResult(
                score=0.0,
                feedback=f"sol.csv must contain columns with 'actual' and 'predicted' in their names. Found columns: {list(predictions_df.columns)}",
                details={"error": "missing_required_columns", "columns": list(predictions_df.columns)}
            )
        
        # Load ground truth data
        nba_data = pd.read_csv('nba_player_stats.csv')
        total_rows = len(nba_data)
        
        # Calculate expected test set size (20% of total)
        expected_test_size = int(0.2 * total_rows)
        actual_test_size = len(predictions_df)
        # Check if test set size is approximately correct (allow some tolerance)
        size_tolerance = 2  # Allow ±2 rows tolerance
        size_score = 1.0 if abs(actual_test_size - expected_test_size) <= size_tolerance else 0.5
        subscores['test_size_correct'] = size_score
        
        if size_score < 1.0:
            feedback_parts.append(f"Test set size: {actual_test_size}, expected: ~{expected_test_size}")
        
        # Load the expected test set (last 20% of rows)
        test_start_idx = int(0.8 * total_rows)
        print(test_start_idx)
        expected_test_data = nba_data.iloc[test_start_idx:].copy()
        expected_actuals = expected_test_data.loc[1:,'Pos'].values  
        
        # Check if the actual values match the expected test set
        actual_values = predictions_df[actual_col].values
        predicted_values = predictions_df[predicted_col].values
        print(len(actual_values), len(expected_actuals))
        # Verify the actual values correspond to the correct test set
        if len(actual_values) == len(expected_actuals):
            actual_match_score = 1.0 if np.array_equal(actual_values, expected_actuals) else 0.0
        else:
            actual_match_score = 0.0
        
        subscores['correct_test_split'] = actual_match_score
        
        if actual_match_score == 0.0:
            feedback_parts.append("The 'actual' values don't match the expected test set (last 20% of rows)")
        
        # Calculate classification accuracy
        if len(actual_values) == len(predicted_values) and len(actual_values) > 0:
            correct_predictions = np.sum(actual_values == predicted_values)
            accuracy = correct_predictions / len(actual_values)
            classification_error = 1.0 - accuracy
            accuracy_score = accuracy  # Score is 1.0 - classification_error = accuracy
            
            subscores['classification_accuracy'] = accuracy_score
            details['accuracy'] = accuracy
            details['classification_error'] = classification_error
            details['correct_predictions'] = int(correct_predictions)
            details['total_predictions'] = len(actual_values)
            
            feedback_parts.append(f"Classification accuracy: {accuracy:.3f}")
            feedback_parts.append(f"Classification error: {classification_error:.3f}")
        else:
            accuracy_score = 0.0
            subscores['classification_accuracy'] = 0.0
            feedback_parts.append("Could not calculate accuracy due to mismatched array lengths")
        
        # Define weights for different components
        weights = {
            'test_size_correct': 0.2,      # 20% for correct test set size
            'correct_test_split': 0.3,     # 30% for using correct test split
            'classification_accuracy': 0.5  # 50% for actual performance
        }
        
        # Calculate final score as weighted average
        final_score = sum(subscores[key] * weights[key] for key in weights.keys())
        
        # Bonus: Give perfect score if all components are perfect
        if all(score >= 0.95 for score in subscores.values()):
            final_score = max(final_score, accuracy_score)
        
        details['test_set_size'] = actual_test_size
        details['expected_test_size'] = expected_test_size
        
        return GradingResult(
            score=final_score,
            subscores=subscores,
            weights=weights,
            feedback=" | ".join(feedback_parts) if feedback_parts else "Task completed successfully!",
            details=details
        )
        
    except Exception as e:
        return GradingResult(
            score=0.0,
            feedback=f"Error processing predictions: {str(e)}",
            details={"error": "processing_error", "exception": str(e)}
        )

#To run locally uncomment below

# if __name__ == "__main__":
#     # Test the grader (useful for debugging)
#     sample_transcript = "Agent executed successfully"
#     result = grade(sample_transcript)
#     print(f"Score: {result.score}")
#     print(f"Feedback: {result.feedback}")
#     print(f"Details: {result.details}")