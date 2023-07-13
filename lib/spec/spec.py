from typing import List, Dict, Optional


class Task:
    """
    Class for defining a task
    pos_examples and neg_examples are straight-forward
    context includes a dictionary that contains the rest of the task-related information (like what it means for customized semantic tags etc.)
    dataset/column indicates the source of the task
    """
    def __init__(self, pos_examples: List[str], neg_examples: List[str], context: Dict, dataset: Optional[str] = None, column: Optional[str] = None, description: Optional[str] = None):
        self.pos_examples: List[str] = [' '.join(pos.split()) for pos in pos_examples]
        self.neg_examples: List[str] = [' '.join(neg.split()) for neg in neg_examples]
        self.context: Dict = context

        self.dataset: Optional[str] = dataset
        self.column: Optional[str] = column
        self.description: Optional[str] = description

    def __repr__(self):
        return 'Task: e+ = [{}], e- = [{}], context= {}'.format(str(self.pos_examples), str(self.neg_examples), self.context)