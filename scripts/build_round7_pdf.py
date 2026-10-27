"""Render the Chinese course report with math and original-size scientific figures."""
from pathlib import Path
import argparse
import hashlib
import html
import re
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def inline(t):
    t=html.escape(t)
    t=re.sub(r'\[([^\]]+)\]\((https?[^)]+)\)',r'<link href="\2" color="#145c7a">\1</link>',t)
    return t.replace('**','').replace('`','')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='docs/round7_report.md')
    p.add_argument('--output',default='docs/round7_report.pdf')
    p.add_argument('--font',default='C:/Windows/Fonts/simsun.ttc')
    args=p.parse_args()
    pdfmetrics.registerFont(TTFont('CN',args.font,subfontIndex=0))
    body=ParagraphStyle('body',fontName='CN',fontSize=10,leading=15.8,wordWrap='CJK',spaceAfter=6)
    title=ParagraphStyle('title',parent=body,fontSize=23,leading=31,textColor=colors.HexColor('#174d60'),spaceAfter=14)
    h1=ParagraphStyle('heading',parent=body,fontSize=15,leading=23,textColor=colors.HexColor('#174d60'),spaceBefore=7,spaceAfter=9,keepWithNext=True)
    h2=ParagraphStyle('subheading',parent=h1,fontSize=11.7,leading=18,spaceBefore=9,spaceAfter=6)
    small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=12.8,spaceAfter=5)
    caption=ParagraphStyle('caption',parent=small,textColor=colors.HexColor('#42565d'),spaceAfter=9)
    source=ROOT/args.source;target=ROOT/args.output
    mathdir=ROOT/'tmp/round7_math';mathdir.mkdir(parents=True,exist_ok=True)
    width=174*mm;story=[];paras=[];rows=[]
    def flush_p():
        if paras:
            story.append(Paragraph(inline(' '.join(paras)),body));paras.clear()
    def flush_t():
        if not rows:return
        n=len(rows[0]);weights=[1]*n
        if n==2:weights=[.31,.69]
        if n==5:weights=[.22,.13,.13,.26,.26]
        cells=[[Paragraph(inline(c),small) for c in row] for row in rows]
        tab=Table(cells,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#7192a0')),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d1dee2')),
            ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.extend([tab,Spacer(1,6)]);rows.clear()
    for line in source.read_text(encoding='utf-8').splitlines():
        if line.strip()=='<!-- pagebreak -->':
            flush_p();flush_t();story.append(PageBreak());continue
        if line.startswith('|'):
            flush_p()
            if not re.fullmatch(r'[| :\-]+',line):rows.append([v.strip() for v in line.strip('|').split('|')])
            continue
        flush_t()
        if not line.strip():flush_p();continue
        if line.startswith('#'):
            flush_p();level=len(line)-len(line.lstrip('#'))
            story.append(Paragraph(inline(line.lstrip('#').strip()),title if level==1 else h1 if level==2 else h2));continue
        if line.startswith('$$') and line.endswith('$$'):
            flush_p();formula=line[2:-2]
            path=mathdir/(hashlib.sha256(formula.encode()).hexdigest()[:16]+'.png')
            math_to_image('$'+formula+'$',path,dpi=600,format='png',prop=FontProperties(size=12),color='#152f3a')
            im=Image(str(path));natural=im.imageWidth/600*72
            scale=min(1.,width/natural);im.drawWidth=natural*scale;im.drawHeight=im.imageHeight/600*72*scale
            im.hAlign='CENTER';story.extend([Spacer(1,4),im,Spacer(1,10)]);continue
        match=re.fullmatch(r'!\[(.*)\]\(([^)]+)\)',line)
        if match:
            flush_p();im=Image(str(source.parent/match[2]));w=(130 if 'strong_baselines' in match[2] else 174 if 'transformer_generalization' in match[2] else 170)*mm
            im.drawHeight*=w/im.drawWidth;im.drawWidth=w;im.hAlign='CENTER'
            story.extend([im,Spacer(1,4),Paragraph(inline(match[1]),caption)]);continue
        paras.append(line)
    flush_p();flush_t()
    def page(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d5e1e6'));canvas.setLineWidth(.5)
        canvas.line(18*mm,17*mm,192*mm,17*mm);canvas.setFont('CN',8);canvas.setFillColor(colors.HexColor('#50626a'))
        canvas.drawString(18*mm,12*mm,'课题12 / Round 7 / 2026-10-27 / AI辅助研究记录')
        canvas.drawRightString(192*mm,12*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(target),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,
        topMargin=16*mm,bottomMargin=22*mm,title='条件Transformer对未见横场的泛化：第七轮',
        author='CTQS project; Codex AI-assisted research record')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(target)


if __name__=='__main__':main()
