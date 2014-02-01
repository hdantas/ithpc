#VARIABLES
TEX    = pdflatex
TMPEXT = *.lof *.lot *.toc *.aux *.out *.log

OUTDIR = out
AUXDIR = auxi

# needed when dir has same name as target
all: lab-report

# PDF GENERATION
lab-report:
	$(TEX) -output-directory=$(OUTDIR) -aux-directory=$(AUXDIR) --jobname="ItHPC Lab Report" $@.tex


# CLEANING UP
clean: clean-miktex clean-tex
	
clean-miktex:
	cd $(AUXDIR) && rm -fv $(TMPEXT)

clean-tex:
	cd $(OUTDIR) && rm -fv $(TMPEXT)
