# KMITL Thesis — Thai YouTube Virality Prediction

This directory contains the LaTeX source of the special-problem thesis for
the Applied Statistics & Data Analytics programme, School of Science, KMITL.

## Files

```
docs/thesis/
├── main.tex                  # XeLaTeX master file
├── references.bib            # BibLaTeX (biber backend)
├── README.md                 # this file
├── frontmatter/              # cover, approval, abstracts, acknowledgement, abbreviations
├── chapters/                 # 01-introduction → 05-conclusion
├── appendix/                 # A-configs / B-tables / C-code / D-reproducibility
├── figures/                  # PNG figures copied from reports/figures/
└── fonts/                    # (optional) drop THSarabunNew.ttf here for proper Thai
```

## How to compile

```bash
cd docs/thesis
xelatex main.tex
biber   main
xelatex main.tex
xelatex main.tex
```

In Overleaf: set the compiler to **XeLaTeX** (Menu → Compiler → XeLaTeX),
upload the whole `docs/thesis/` folder, and add the four
`THSarabunNew*.ttf` files into `fonts/` (auto-fallback to Norasi otherwise).

## Where each number comes from

Every numerical claim in `chapters/04-results.tex` traces back to a CSV under
`reports/tables/` and a parquet under `reports/artifacts/predictions/`.
Re-run from scratch with `make prepare && make eval` from the repository
root; MLflow run UI under `mlruns/` is the source of truth.

## TH Sarabun New (optional but recommended)

Download from <https://www.f0nt.com/release/th-sarabun-new/> and drop the
four `.ttf` files into `docs/thesis/fonts/` exactly as:

```
THSarabunNew.ttf
THSarabunNew-Bold.ttf
THSarabunNew-Italic.ttf
THSarabunNew-BoldItalic.ttf
```

If absent, `main.tex` auto-falls back to **Norasi** (ships with TeX Live).

## License & Credit

ผลงานทั้งหมดของปัญหาพิเศษเล่มนี้ ทั้งเล่มวิทยานิพนธ์ รหัสฐาน ระเบียบวิธีวิจัย
ผลการทดลอง และคำอธิบายต่าง ๆ เป็นเครดิตของนิสิตทั้งสามคน

- นายตรัยภูรินท์ สืบสุวรรณสาร (รหัสนักศึกษา 65050330)
- นายธีรนนท์ ไชยลังกา (รหัสนักศึกษา 65050417)
- นายภาณุเดช ภู่โทสนธิ์ (รหัสนักศึกษา 65050685)

อาจารย์ที่ปรึกษา: ผศ. ดร. พรพิมล ชัยวุฒิศักดิ์
ภาควิชาสถิติประยุกต์และการวิเคราะห์ข้อมูล คณะวิทยาศาสตร์
สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง · ปีการศึกษา 2568
