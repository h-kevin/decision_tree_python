import copy
import sys
import math
import random

class Example:
    """
    A class representing an example.
    An example has an attributes instance which stores the values
    of the example and a classification instance to store the
    example's classification.
    """
    def __init__(self):
        # key - value pairs of the attribute name and its
        # corresponding value of this example.
        self.attributes = {}
        
        # the category the example belongs to.
        self.classification = ""

        
class Node:
    """
    A node represents the attribute we want to split
    the examples on in the Decision Tree.
    """

    def __init__(self, value):
        # the value of the node can either the next
        # attribute to split the examples on or the
        # classification if the node is a leaf
        self.value = value
        
        self.children = []
        self.label = "" # the value of the attribute, if applicable.
        
    def is_leaf(self):
        """
        Return True if the current node is a leaf,
        False otherwise.
        """
        return len(self.children) == 0


class DecisionTree:
    """
    A decision tree, based on the decision tree algorithm.
    """

    # value to return in case the decision tree fails
    # to classify the example.
    DEFAULT_VALUE = "no"

    # Construct a DecisionTree. The root node of the tree 
    # is initialized to contains the specified attribute.
    def __init__(self, attribute):
        root = Node(attribute)
        self.root = root
        self.size = 1
        
    def add_branch(self, t):
        """
        Add the subtree t to the root of the tree.
        """
        self.root.children.append(t.root)
        self.size += t.size
    
    def classify(self, e):
        """
        Return the predicted classification of the example e
        based on the training examples and the decision tree 
        learning algorithm.
        """
        return self.classify_helper(e, self.root)
    
    def classify_helper(self, e, root):
        """
        Helper function for classify.
        """
        # return the classification of the example e based
        # on the DecisionTree.
        if root.is_leaf():
            return root.value
        else:
            attr = root.value
            value = e[attr]
            
            for node in root.children:
                if node.label == value:
                    return self.classify_helper(e, node)
            
            return self.DEFAULT_VALUE
            
    def print_tree(self):
        """
        Print all the nodes of the tree in a readable format.
        """
        self.print_tree_helper(self.root, 0)
                
    def print_tree_helper(self, root, depth):
        """
        Helper function for print_tree.
        """
        if root.is_leaf():
            print(("\t" * depth) + root.value)
        else:
            for child in root.children:
                print("{} {} = {}:".format("\t" * depth, root.value,
                        child.label))
                self.print_tree_helper(child, depth + 1)
                
def decision_tree_learning(examples, attributes, parent_examples):
    """
    Return a DecisionTree constructed based on a list of training examples and a list of attributes.
    Based on AIMA 3rd edition, p. 702.
    """
    if not examples:
        return DecisionTree(plurality_value(parent_examples))
    elif same_classification(examples):
        return DecisionTree(examples[0].classification)
    elif not attributes:
        return DecisionTree(plurality_value(examples))
    else:
        A = max(
            attributes.keys(), 
            key = lambda a: importance(a, examples, attributes)
            )
        tree = DecisionTree(A)
        
        for value in attributes[A]:
            exs = [e for e in examples if e.attributes[A] == value]
            
            new_attributes = copy.copy(attributes)
            new_attributes.pop(A, None)
            subtree = decision_tree_learning(exs, new_attributes, examples)
            
            # add a branch to tree with label (A = value) and 
            # subtree "subtree"
            subtree.root.label = value
            tree.add_branch(subtree)
            
        return tree
            
def get_examples(input):
    """
    Given the input, return a list of examples built from that input.
    """
    examples = []

    # get the names of the attributes, which are specified in 
    # the first line of input except for the last value.
    attrs = input[0][:-1]

    # The rest of the input are the values for the attributes
    # and the corresponding classification, so parse the input to
    # create examples.
    for i in range(1, len(input)):
        current_example = input[i]
        e = Example()

        for j in range(len(attrs)):
            e.attributes[attrs[j]] = current_example[j]

        classification = current_example[-1]
        if classification[-1] == "\n": # remove newline char left when parsing
            classification = classification[:-1]
        e.classification = classification
        
        examples.append(e)

    return examples

def get_attributes(input):
    """
    Given the input, return a dictionary of attributes built 
    from that input. The keys are the names of the attributes, and
    the values are lists of all the possible values the attributes
    can have.
    """
    result = {}
    
    # get the names of the attributes, which are specified in 
    # the first line of input except for the last value.
    attrs = input[0][:-1]
    
    for i in range(len(attrs)):
        attr_name = attrs[i]
        result[attr_name] = []

        # parse each line, and add any new value corresponding
        # to attr_name.
        for j in range(1, len(input)):
            value = input[j][i]
            if not value in result[attr_name]:
                result[attr_name].append(value)

    return result

def parse_input(filename):
    """
    Take the name of the file which contains the examples
    and return a list of list to parse information more efficiently
    """
    input = open(filename, "r").readlines()
    result = [line.split("\t") for line in input]
    return result
    
def get_examples_and_attributes(filename):
    """
    Take the name of the input file and return a list of 
    examples and attributes parsed from that file.
    """
    input = parse_input(filename)
    examples = get_examples(input)
    attributes = get_attributes(input)
    return (examples, attributes)
    
def importance(a, examples, attributes):
    """
    Return the gain from splitting the tree with attribute a.
    Based on AIMA, 3rd edition p. 703.
    """
    
    # Calculate the entropy of the parent set.
    number_of_positives = len([e for e in examples 
            if e.classification[0] == "y"])
    b = entropy(number_of_positives / len(examples))
            
    # Calculate the remainder.
    remainder = 0
    for value in attributes[a]:
        p, n = get_positives_and_negatives_of_value(a, value, examples)
        if p + n != 0:
            remainder += (p + n) / len(examples) * entropy(p / (p + n))
        
    gain = b - remainder
    
    return gain
    
def get_positives_and_negatives_of_value(a, value, examples):
    """
    Return the number of positive and negative examples whoses attribute
    a has the value value in the given list of examples.
    """
    positives = [e for e in examples if e.classification[0] == "y"
            and e.attributes[a] == value]
    negatives = [e for e in examples if e.classification[0] == "n"
            and e.attributes[a] == value]
    
    return (len(positives), len(negatives))

def entropy(q):
    """
    Given a probability q, return its entropy.
    """
    if q == 1 or q == 0: return 0.0
    return -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))
    
def plurality_value(examples):
    """
    Return the category that has the highest number of examples.
    """
    categories = {} # mapping of a category to its number of appearances
                    # in the examples.
    
    for e in examples:
        classification = e.classification
        if not classification in categories.keys():
            categories[classification] = 1
        else:
            categories[classification] += 1
    
    # get the category that has the highest number of appearances.
    max_value = -sys.maxsize
    max_category = None
    
    for c in categories:
        value = categories[c] + random.random()
        if value > max_value:
            max_value = value
            max_category = c
    
    return max_category
    
def same_classification(examples):
    """
    Return True if all the examples have the same classification,
    False otherwise.
    """
    if len(examples) == 0: return True
    
    # classification of the first example.
    classification = examples[0].classification
    
    # compare the classification of the first example 
    # with the rest of the examples.
    for e in examples:
        if e.classification != classification:
            return False
            
    return True
    
def leave_one_out_cross_validation(filename):
    """
    Take in the name of the input file.
    Return the accuracy value using leave-one-out cross-validation.
    """
    examples, attributes = get_examples_and_attributes(filename)
    number_of_examples = len(examples)
    
    correct_guesses = 0
    
    for i in range(number_of_examples):
        test_examples = copy.copy(examples)
        e = test_examples.pop(i)
        
        t = decision_tree_learning(test_examples, attributes, 
                test_examples)
        if t.classify(e.attributes) == e.classification:
            correct_guesses += 1
        
    return correct_guesses / number_of_examples
        
    
def get_accuracy(tree, examples):
    """
    Take in the tree and the examples.
    Return the accuracy of the training set.
    """
    correct_guesses = 0
    
    for e in examples:
        if tree.classify(e.attributes) == e.classification:
            correct_guesses += 1
    
    return correct_guesses / len(examples)