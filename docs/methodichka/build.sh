#!/bin/zsh
# Собирает методичку: Markdown → HTML (MathML, pandoc) → PDF (headless Chrome).
# Нужны pandoc ≥ 3 и Google Chrome. Рисунки: python figures/chNN_figures.py.
set -e
cd "${0:A:h}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p build
cat > build/cover.html <<'HTML'
<div class="cover">
<div class="kicker">Методическое пособие · проект morse-lidar</div>
<div class="big">Топология и геометрия 3D-поверхностей для дескриптора Морса–Смейла</div>
<div class="sub">Теория Морса, триангуляции, персистентные гомологии, кривизна и граничные эффекты — ровно то, что нужно, чтобы понимать и менять код проекта</div>
<div class="meta">Для команды проекта: Чупров Д. А., Евстратов И. М.<br>РТУ МИРЭА, ИПТИП · 2026</div>
</div>
HTML
pandoc ch00.md ch0[1-9].md ch1[01].md \
  --from markdown+tex_math_dollars+pipe_tables+implicit_figures \
  --standalone --toc --toc-depth=2 --math-method=mathml \
  --metadata title="Методичка morse-lidar" --metadata lang=ru --metadata toc-title="Содержание" \
  --css style.css --include-before-body=build/cover.html --embed-resources --resource-path=.:figures \
  -o build/metodichka.html
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=methodichka.pdf \
  "file://$PWD/build/metodichka.html" 2>/dev/null
