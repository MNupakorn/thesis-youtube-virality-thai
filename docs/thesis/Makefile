# docs/thesis/Makefile — บังคับ compile chain ที่ถูกต้อง
# วิธีใช้: make            (สร้าง main.pdf)
#         make clean      (ลบไฟล์ชั่วคราว)
#         make distclean  (ลบรวม main.pdf)

MAIN = main

.PHONY: all clean distclean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib kmitl-stat.bbx
	xelatex -interaction=nonstopmode $(MAIN).tex
	biber $(MAIN)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex

# ใช้ latexmk (สะดวกกว่า) ถ้ามีในระบบ
latexmk:
	latexmk -xelatex $(MAIN).tex

watch:
	latexmk -xelatex -pvc $(MAIN).tex

clean:
	rm -f *.aux *.log *.bbl *.bcf *.blg *.lof *.lot *.out *.run.xml *.synctex.gz *.toc

distclean: clean
	rm -f $(MAIN).pdf
