from decision_tree import *

def main(filename):
    """The main function that takes in the input file name
    and prints out the tree."""
    examples, attributes = get_examples_and_attributes(filename)

    # create a decision tree based on the examples.
    t = decision_tree_learning(examples, attributes, examples)
    
    # print the tree
    t.print_tree()

if __name__ == "__main__":
    main(sys.argv[1])