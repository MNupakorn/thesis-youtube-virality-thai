# .latexmkrc — บังคับให้ latexmk ใช้ xelatex + biber
# ใช้คำสั่ง: latexmk -xelatex main.tex
$pdf_mode = 5;          # ใช้ xelatex สร้าง PDF
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;        # รัน biber/bibtex ทุกครั้งที่จำเป็น
$biber = 'biber --validate-datamodel %O %S';
$clean_ext = 'bbl run.xml synctex.gz';
