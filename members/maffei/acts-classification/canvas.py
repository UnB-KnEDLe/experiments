from drawBoxes.drawBoxes import draw
import sys

import fitz
import argparse

parser = argparse.ArgumentParser(description="draw PDFs boudingboxes")

parser.add_argument('file_lis',
    metavar='FILE',
    help='paths to pdfs', 
    nargs='+',
)
parser.add_argument('--i', help='whether draw image bounding boxes',
    type=int, default=1)
parser.add_argument('--w', help='whether draw word bounding boxes',
    type=int, default=0)
parser.add_argument('--l', help='whether draw line bounding boxes',
    type=int, default=0)
parser.add_argument('--t', help='whether draw text block bounding boxes',
    type=int, default=1)
args = parser.parse_args()

for fname in args.file_lis:
    outfile = fname.split('.pdf')[0]+'_canvas_i{}_w{}_l{}_t{}.pdf'.format(
        args.i, args.w, args.l, args.t, 
    )
    draw(
        fitz.open(fname),
        img=args.i,
        word=args.w,
        line=args.l,
        txt=args.t,
    ).save(outfile)

    print("Saved canvas of {} at >> {}".format(fname,outfile))

