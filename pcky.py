#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-

"""An implementation of the CKY parsing algorithm for probabilistic CFGs."""

import numpy as np
from nltk import PCFG, Production, Tree

class PCKYParser(object):
    """A Cocke-Younger-Kasami (CKY or CYK) parser that parses 
    according to a probabilistic, context free grammar (PCFG).
    
    This implementation only supports PCFGs in Chomsky-normal-form (CNF).
    I.e., production rules must be binary if non-lexical, or unary if 
    lexical."""
    def __init__(self, grammar):
        super(PCKYParser, self).__init__()
        self.grammar = self.load_grammar(grammar)
        self.index = CodeBook(
            set(map(Production.lhs, self.grammar.productions()))
        )
    
    def load_grammar(self, grammar_path):
        """Returns a PCFG from the specified file."""
        with open(grammar_path, 'r') as f:
            pcfg = PCFG.fromstring(f.read())
            if not pcfg.is_chomsky_normal_form():
                raise ValueError("grammar not in Chomsky normal form")
            return pcfg
    
    def terminals(self):
        """Returns productions of the form A -> word."""
        return (
            p for p in self.grammar.productions()
            if Production.is_lexical(p)
        )
    
    def nonterminals(self):
        """Returns productions of the form A -> B C."""
        return (
            p for p in self.grammar.productions()
            if Production.is_nonlexical(p)
        )
        
    def parse(self, words):
        """Parses a sequence of words according to the CKY algorithm.
         
        This method returns the single parse tree with the highest probability 
        according to the PCFG that covers the input.  If the input is not 
        grammatical, an empty tree is returned."""
        matrix_shape = (len(words)+1, len(words)+1, len(self.index))
        matrix = np.zeros(matrix_shape)
        back_pointer_matrix = np.empty(matrix_shape, dtype=object)
        # Dynamically populate the probability and back-pointer matrices
        for j in range(1, len(words)+1):
            word = words[j-1]
            # Fill lexical production entries (along the diagonal)
            for production in self.grammar.productions(rhs=word):
                a = self.index[production.lhs()]
                matrix[j-1,j,a] = production.prob()
                back_pointer_matrix[j,j,a] = production.rhs()[0]
            # Fill non-lexical production entries (above the diagonal)
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for production in self.nonterminals():
                        a, b, c = map(
                            self.index.get,
                            (production.lhs(),) + production.rhs()
                        )
                        left, right = matrix[i,k,b], matrix[k,j,c]
                        score = production.prob() * left * right
                        # Maximize the parse probability
                        if matrix[i,j,a] < score:
                            matrix[i,j,a] = score
                            back_pointer_matrix[i,j,a] = (k,) + production.rhs()
        return self.build_tree(
            back_pointer_matrix,
            0,
            len(words),
            self.grammar.start()
        )
    
    def build_tree(self, back, row, col, root):
        """Given a back-pointer matrix, a row/column entry point into the
        back-pointer matrix, and the root label, recursively builds and returns 
        the most probable syntactic parse tree rooted at the entry point."""
        a = self.index[root]
        # Base case - lexical productions
        if root in set(map(Production.lhs, self.terminals())):
            return Tree(root, [back[row+1,row+1,a]])
        # Recursive case - nonlexical productions
        else:
            try:
                k, b, c = back[row,col,a]
                left, right = [back,row,k,b], [back,k,col,c]
                return Tree(
                    root,
                    [self.build_tree(*left), self.build_tree(*right)]
                )
            except TypeError:
                # In case the input is unlicensed by the PCFG
                return Tree(None, [])
            except Exception as e:
                raise e

class CodeBook(object):
    """A bi-directional map between names and auto-generated indices."""
    
    def __init__(self, names):
        self.names = dict((index, name) for index, name in enumerate(names))
        self.index = dict((name, index) for index, name in enumerate(names))
    
    def __contains__(self, name):
        return name in self.index
    
    def __getitem__(self, name):
        return self.index[name]
    
    def __iter__(self):
        return iter(self.index)
    
    def __len__(self):
        return len(self.index)
    
    def __repr__(self):
        return "<%s with %d entries>" % (self.__class__.__name__, len(self))
    
    def add(self, name):
        """Add the given name with a generated index."""
        if name not in self:
            index = len(self)
            self.names[index] = name
            self.index[name] = index
        return name
    
    def get(self, name, default=None):
        """Return the index associated with the given name."""
        return self.index.get(name, default)
    
    def name(self, index):
        """Return the name associated with the given index."""
        return self.names[index]
