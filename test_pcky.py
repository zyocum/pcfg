#!/usr/bin/env python3

import sys
#from string import split
from unittest import TestCase, main, skip

from nltk.probability import ImmutableProbabilisticMixIn as IPMI
from nltk.grammar import Nonterminal

from pcky import PCKYParser

class PCKYParserTest(TestCase):
    """Tests for the probabilistic CKY parser."""
    
    def setUp(self):
        """Set up sentences, grammars, and parsers."""
        # These sentences present syntactic ambiguity in terms of PP attachment
        with open('sentences.txt', mode='r') as f:
            sentences = f.readlines()
        self.sentences = [s.split() for s in sentences]
        self.grammars = (
           #'non_cnf.pcfg',  # A trivial non-CNF PCFG with one trinary rule
           #'trivial.pcfg',  # A trivial CNF PCFG that licenses a unique input
            'grammar1.pcfg', # A toy grammar whose PPs prefer attaching to NPs
            'grammar2.pcfg'  # A toy grammar whose PPs prefer attaching to VPs
        )
        self.parsers = [PCKYParser(g) for g in self.grammars]
    
    def most_probable(self, lhs, grammar):
        return max(grammar.productions(lhs=lhs), key=IPMI.prob)
    
    def test_non_cnf(self):
        """Test loading incompatible grammar"""
        try:
            PCKYParser('non_cnf.pcfg')
        except ValueError as e:
            print('message:', e, file=sys.stderr)
            self.assertEqual(str(e), "grammar not in Chomsky normal form")
        
    def test_trivial_grammar(self, verbose=sys.stderr):
        """Test trivial PCFG"""
        parser = PCKYParser('trivial.pcfg')
        licensed_words = 'left right'.split()
        unlicensed_words = 'right left'.split()
        licensed = str(parser.parse(licensed_words))
        unlicensed = str(parser.parse(unlicensed_words))
        if verbose:
            print('', file=verbose)
            print('{} -> {}'.format(licensed_words, licensed), file=verbose)
            print('{} -> {}'.format(unlicensed_words, unlicensed), file=verbose)
        # Check that the licensed and unlicensed words produce appropriate trees
        self.assertEqual(licensed, '(A (B left) (C right))')
        self.assertEqual(unlicensed, '(None )')
    
    def test_syntactic_ambiguity(self, verbose=sys.stderr):
        """Test toy PCFGs with syntactically ambiguous sentence"""
        np, pp, vp, v, verb = (
            Nonterminal(l) for l in('NP', 'PP', 'VP', 'V', 'Verb')
        )
        grammars = [p.grammar for p in self.parsers]
        # Check that the grammars differ in their VP rule probabilities
        self.assertEqual(self.most_probable(vp, grammars[0]).rhs(), (verb, np))
        self.assertEqual(self.most_probable(vp, grammars[1]).rhs(), (v, pp))
        # There are two readings for the sentence
        sentence = self.sentences[0]
        # The readings are represented by the following trees
        readings = (
            """(S
  (NP (Det the) (Noun defendant))
  (VP
    (Verb hit)
    (NP
      (NP (Det the) (Noun lawyer))
      (PP (Prep with) (NP (Det the) (Noun briefcase))))))""",
            """(S
  (NP (Det the) (Noun defendant))
  (VP
    (V (Verb hit) (NP (Det the) (Noun lawyer)))
    (PP (Prep with) (NP (Det the) (Noun briefcase)))))"""
        )
        if verbose:
            for i, parser in enumerate(self.parsers, 1):
                grammar = parser.grammar
                rule = "\nMost probable VP rule for grammar{} : {}"
                print(rule.format(i, self.most_probable(vp,grammar)), file=verbose)
                tree = "\nMost probable parse tree for grammar{} :\n{}"
                print(tree.format(i, parser.parse(sentence)), file=verbose)
        # Check that grammar1 produces reading0 and grammar2 produces reading1
        # string representation comparison to determine isomorphism
        for i in range(len(readings)):
            self.assertEqual(str(self.parsers[i].parse(sentence)), readings[i])

if __name__ == '__main__':
    # Run the tests, print the results, and exit.
    main(verbosity=2)
