class Value:
    """
        Class representing a node in a computation graph
    """
    def __init__(self, data, children = (), op='', label=''):
        """
            Constructor for the Value class

            Parameters:
                - data (float): data stored in the Value instance
                - children (tuple, optional): tuple of Value instances that are the inputs to this Value instance in the computation graph. Default is an empty tuple.
                - op (str, optional): string that represents the operation performed to compute the value stored in this Value instance. Default is an empty string.
                - label (str, optional): string for labeling the Value instance in the computation graph. Default is an empty string.

            Attributes:
                - data (float): data stored in the Value instance
                - _prev (set): set of Value instances that are the inputs to this Value instance in the computation graph
                - _op (str): string that represents the operation performed to compute the value stored in this Value instance
                - label (str): string for labeling the Value instance in the computation graph
                - backprop (function): lambda function that performs the backpropagation for this Value instance
                - grad (float): gradient of this Value instance, initialized to 0.0
        """
        self.data = data
        self._prev = set(children)
        self._op = op
        self.label = label
        self.backprop  = lambda: None
        self.grad = 0.0
    
    def _create_other(self, other):
        """
            Helper function to convert input to a Value instance if it's not already one

            Parameters:
                - other (float or Value): value to be converted to a Value instance if needed

            Returns:
                - Value: other converted to a Value instance if it was not already one
        """
        return other if isinstance(other, Value) else Value(data=other)
        
    def __add__(self, other):
        """
            Overloads the '+' operator for the Value class

            Parameters:
                - other (float or Value): right-side operand in the addition operation

            Returns:
                - Value: result of the addition operation
        """
        other = self._create_other(other)
        out = Value(self.data + other.data, children=(self, other), op='+')
        def _backprop():
            self.grad += out.grad
            other.grad += out.grad
        out.backprop = _backprop
        return out

    def __mul__(self, other):
        """
            Overloads the '*' operator for the Value class

            Parameters:
                - other (float or Value): right-side operand in the multiplication operation

            Returns:
                - Value: result of the multiplication operation
        """
        other = self._create_other(other)
        out = Value(self.data * other.data, children=(self, other), op='*')
        def _backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backprop = _backprop 
        return out
    
    
    def __pow__(self, other):
        """
            Overrides the power operator.
            
            Args:
                - other: The power to raise `self` to
            
            Returns:
                - The result of the power operation as a new `Value` node
        """
        out = Value(self.data**other, (self,))
        def _backprop():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out.backprop = _backprop
        return out
    
    
    def backward(self):
        """
            Backpropagates the gradient through the computational graph.
        """
        sorted_graph = []
        visits_node = set()
        def _build(node):
            if node not in visits_node:
                visits_node.add(node)
                for children in node._prev:
                    _build(children)
                sorted_graph.append(node)
        _build(self)
        self.grad = 1.0
        for node in reversed(sorted_graph):
            node.backprop()
    
    
    def __sub__(self, other):
        """
            Overrides the subtraction operator (self - other)
            Returns the result of the subtraction as a new Value object
        """
        return self + (-other)
    
    def __rsub__(self, other):
        """
            Overrides the subtraction operator (other - self)
            Returns the result of the subtraction as a new Value object
        """
        return other + (-self)
    
   
    def __radd__(self, other):
        """
            Overrides the addition operator (other + self)
            Returns the result of the addition as a new Value object
        """
        return self + other

    def __rmul__(self, other):
        """
            Overrides the multiplication operator (other * self)
            Returns the result of the multiplication as a new Value object
        """
        return self * other
    
    def __truediv__(self, other):
        """
            Overrides the true division operator (self / other)
            Returns the result of the division as a new Value object
        """
        return self * other**-1
    
    def __rtruediv__(self, other):
        """
            Overrides the true division operator (other / self)
            Returns the result of the division as a new Value object
        """
        return other * self**-1
        
    def __neg__(self):
        """
            Overrides the negation operator (-self)
            Returns the negated value as a new Value object
        """
        return self * -1
    
    
    def relu(self):
        """
            Computes the ReLU activation function
            Returns a new Value object representing the result
        """
        return Value(max(self.data, 0), children=(self,), op='relu')
    
                        
    def __repr__(self):
        """
            Returns a string representation of the Value object
        """
        return f"Value(data={self.data}, fn={self.backprop})"