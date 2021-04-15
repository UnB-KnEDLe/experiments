import re

class TTerm:
    name = ''
    begin = []
    end  = []
    
#Constructor: Update the following properties: 
#             name - String of a term 
#             begin, end - term positions related to input parameter ptext 

    def __init__(self, ptext, pterm):
        self.begin = []
        self.end = []
        self.name = pterm
        pstart = 0
        while (pstart < len(ptext)):
            paux = re.search(pterm, ptext[pstart:])
            if paux is not None:
                pauxbegin, pauxend = paux.span()
                self.begin.append(pauxbegin + pstart)
                self.end.append(pauxend + pstart)
                pstart     = pauxend + pstart
            else: 
                break;

class TExpression:
    name = ''
    terms = []
    
#Constructor: Update the following properties: 
#             name - String of a Regular Expression
#             terms - list of terms found by searching for a RE (name).

    def __init__(self, ptext, pexpression):
        self.name = pexpression
        pauxterms = re.findall(self.name, ptext) #finding all terms related to one RE (pexpression).
        pauxterms = set(pauxterms) # list with unique terms
        self.terms = []
        for pt in pauxterms:
            self.terms.append(TTerm(ptext, pt))
            
class TREClass:
    token = ''
    expressionnamelist = []
    expressions = []
    
#Constructor: Update the following properties: 
#             expressionnamelist - it must hold a list of Regular Expressions.
#             token: it represents all REs in expressionnamelist

    def __init__(self, ptoken, pexpressionnamelist):
        self.token = ptoken
        self.expressionnamelist = pexpressionnamelist
        
#Update the propertie expressions - it must hold the found terms inside the input parameter ptext.
    def applyExpressions(self, ptext): #applying the set of RE on a text.
        self.expressions = []
        for pexp in self.expressionnamelist:
            self.expressions.append(TExpression(ptext, pexp))
    
#Return: A dictionary with regular expressions and its unique terms related to the processed text (after applyExpressions)    
    def getExpressions(self):
        pret = {}
        for pexp in self.expressions:
            pret[pexp.name] = [] #list of terms 
            for pt in pexp.terms:  #Terms of the expression pexp
                if pt.name not in pret[pexp.name]: 
                    pret[pexp.name].append(pt.name)
        return pret

#Return: A dictionary with unique terms and their locations related to the processed text (after applyExpressions)    
    def getTerms(self):
        pret = {}
        for pexp in self.expressions:
            for pt in pexp.terms:
                if pt.name not in pret: 
                    pret[pt.name] = []
                pret[pt.name].append((pt.begin, pt.end))
        return pret

#Return: A score related to the processed text (after applyExpressions): 
#        Proportional to    
    def getScore(self):
        pret = 0
        for pexp in self.expressions:
            for pt in pexp.terms:
                psize = len(pt.name.split()) #term size (in words)
                pfreq = len(pt.begin) # How many times this term occurs
                pret  = pret + psize*pfreq
        return (pret)

