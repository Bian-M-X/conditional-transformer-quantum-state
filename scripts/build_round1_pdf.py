"""Render the evidence-backed Chinese round report; requires reportlab and a CJK font."""
import argparse
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]


def inline(text):
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<link href="\2" color="#175d8c">\1</link>',text)
    return text.replace('**','').replace('`','')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--font',default='C:/Windows/Fonts/simsun.ttc')
    parser.add_argument('--source',default='docs/round1_report.md')
    parser.add_argument('--output',default='docs/round1_report.pdf')
    parser.add_argument('--footer',default='CTQS / Round 1 / 2026-10-19 / 初筛研究报告')
    parser.add_argument('--title',default='条件 Transformer 量子态：第一轮机制与泛化研究')
    args=parser.parse_args()
    pdfmetrics.registerFont(TTFont('CJK',args.font,subfontIndex=0))
    styles=getSampleStyleSheet()
    normal=ParagraphStyle('CN',fontName='CJK',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK')
    title=ParagraphStyle('TitleCN',parent=normal,fontSize=21,leading=29,textColor=colors.HexColor('#174d60'),spaceAfter=16)
    h1=ParagraphStyle('H1CN',parent=normal,fontSize=14,leading=21,spaceBefore=14,spaceAfter=8,textColor=colors.HexColor('#174d60'),keepWithNext=True)
    h2=ParagraphStyle('H2CN',parent=h1,fontSize=11.5,leading=18)
    tiny=ParagraphStyle('TinyCN',parent=normal,fontSize=8,leading=12,spaceAfter=0)
    source=ROOT/args.source
    target=ROOT/args.output
    story=[]; table_rows=[]; paragraph=[]
    width=174*mm
    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)),normal));paragraph.clear()
    def flush_table():
        if not table_rows:return
        count=len(table_rows[0])
        cells=[[Paragraph(inline(c),tiny) for c in row] for row in table_rows]
        table=Table(cells,colWidths=[width/count]*count,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#bac8cd')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([table,Spacer(1,8)]);table_rows.clear()
    for line in source.read_text(encoding='utf-8').splitlines():
        if line.startswith('|'):
            flush_paragraph()
            if re.fullmatch(r'[| :\-]+',line):continue
            table_rows.append([c.strip() for c in line.strip('|').split('|')]);continue
        flush_table()
        if not line.strip():flush_paragraph();continue
        if line.startswith('#'):
            flush_paragraph();level=len(line)-len(line.lstrip('#'))
            story.append(Paragraph(inline(line.lstrip('#').strip()),title if level==1 else h1 if level==2 else h2));continue
        match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            flush_paragraph()
            image=Image(str(source.parent/match[2]))
            image.drawHeight*=width/image.drawWidth;image.drawWidth=width
            story.extend([image,Paragraph(inline(match[1]),tiny),Spacer(1,10)]);continue
        if line.startswith('- '):
            flush_paragraph();story.append(Paragraph(inline('• '+line[2:]),normal));continue
        paragraph.append(line)
    flush_paragraph();flush_table()
    def footer(canvas,doc):
        canvas.setFont('CJK',8);canvas.setFillColor(colors.HexColor('#56656b'))
        canvas.drawString(18*mm,12*mm,args.footer)
        canvas.drawRightString(192*mm,12*mm,str(doc.page))
    doc=SimpleDocTemplate(str(target),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,
        topMargin=17*mm,bottomMargin=20*mm,title=args.title,author='CTQS research project; AI-assisted draft')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)


if __name__=='__main__':
    main()
