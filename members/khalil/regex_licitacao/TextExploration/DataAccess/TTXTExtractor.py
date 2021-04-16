#import docx  - not yet implemented 
import pdfplumber
import os

# Possible Improvements:
# 1. Do we need to add another formats (.docx, csv, txt, ...)? 
#       1.1 At least we need to prepare this to the 
#       DODFMiner output format. 
# 2. Do we need another methods? Like:
#       2.1 Returning one text from all input file pages
#       2.2 Returning one text from a range of pages

class TTXTExtractor:
    fextension = ''
    fname = ''
    filename = ''
    pdf = None
    
    def __init__(self, pfilename):
        auxname, auxextension = os.path.splitext(pfilename)
        if (auxextension == '.pdf'):
            self.fname, self.fextension, self.filename = auxname, auxextension, pfilename
            self.pdf = pdfplumber.open(self.filename)
        
    def getPageText(self, pnumpage):
        PageText = []
        pText = []
        page = self.pdf.pages[pnumpage]
        pText.append(page.extract_text())
        PageText = '\n'.join(pText)
        return PageText
