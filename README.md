# Decision Trees: Concepts and Implementation

This repository explains Decision Trees based on the provided PDF, covering key concepts like entropy, information gain, and Gini impurity for building decision trees in classification tasks.
# In Decision Tree always prefer to give max_depth otherwise it would overfit in most of cases.
## Key Concepts from the PDF

### 1. Basics of Decision Trees
Decision trees work for **categorical and numerical variables**. The process involves:
- Starting with a training dataset (classification/regression output).
- Determining the **best feature** to split the data (root node selection).
- Splitting data into subsets (higher/lower values) for each node.
- Recursively generating new nodes until stopping criteria are met.

### 2. Entropy
Entropy measures **disorder** in the data.  
Formula:  
Entropy = Σ (-p_i * log₂(p_i)) for i = 1 to C
- Higher entropy = more uncertainty (max entropy for balanced classes).  
- Lower entropy = more homogeneity (e.g., one dominant class).  

**Example**:  
For binary classification:  
- If probabilities are 0.8 and 0.2 → lower entropy.  
- If probabilities are 0.5 and 0.5 → maximum entropy.  

### 3. Information Gain
Metric to select the **best feature** for splitting.  
- Measures **reduction in entropy** after a split.  
- Higher gain = better split.  
Formula:  
Info Gain = Entropy(parent) - Σ (weight_child * Entropy(child))


### 4. Gini Impurity
Alternative to entropy for evaluating splits.  
Formula:  
Gini = 1 - Σ (p_i²) for i = 1 to C
- Lower Gini = better split (0 for pure nodes).  

---

## How to Build a Decision Tree (Step-by-Step)

1. **Sort Data**  
   - Arrange numerical features in ascending order.  

2. **Evaluate Splits**  
   - Test every possible split (e.g., `feature > 1.6`).  
   - Calculate **entropy/Gini** for child nodes.  

3. **Select Best Split**  
   - Choose the split with the **highest information gain** or **lowest Gini impurity**.  

4. **Recursive Splitting**  
   - Repeat for subsets until:  
     - All leaf nodes are pure (single class).  
     - Maximum depth is reached.  

---

## Example Code Snippet
```python
def calculate_entropy(probabilities):
    return -sum(p * log(p) for p in probabilities if p > 0)

def information_gain(parent_entropy, children_entropy):
    return parent_entropy - sum(child_weight * child_entropy for child_weight, child_entropy in children_entropy)
