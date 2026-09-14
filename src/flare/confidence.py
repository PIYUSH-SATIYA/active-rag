def calculate_confidence(probabilities: list[float], threshold: float) -> bool:
    """
    Evaluates whether a generated sequence is confident based on token probabilities.
    In the FLARE algorithm, if any token probability drops below the threshold,
    the sequence is considered low-confidence (needs retrieval).
    
    Args:
        probabilities (list[float]): The probability of each generated token.
        threshold (float): The confidence threshold (e.g., 0.8).
        
    Returns:
        bool: True if all probabilities are >= threshold, False otherwise.
    """
    if not probabilities:
        return True
        
    for prob in probabilities:
        if prob < threshold:
            return False
            
    return True
