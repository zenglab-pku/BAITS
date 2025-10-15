
<div align="center">
<img src="https://github.com/zenglab-pku/BAITS/raw/master/docs/_static/BAITS_logo.png" width="400px">

**A Python package for <span style="color:#660974">B</span> cell repertoire <span style="color:#660974">a</span>nalysis and l<span style="color:#660974">i</span>neage <span style="color:#660974">t</span>racking in <span style="color:#660974">s</span>patial omics.**

---

<p align="center">
  <a href="https://baits.readthedocs.io/en/latest/index.html" target="_blank">Documentation</a> •
  <a href="https://github.com/zenglab-pku/BAITS/tree/main#" target="_blank">Github</a>
</p>


## Background

<p>
In contrast to previously reported bulk or single-cell immune receptor sequencing technologies, the simultaneous acquisition of spatially resolved high-dimensional gene expression profiles and complex immune receptor sequences from the same tissue section presents unique challenges for bioinformatic analysis. 
</p>

<p>
To systematically investigate the spatial organization and clonal dynamics of B cells in the tumor microenvironment, we developed BAITS (B cell repertoire Analysis and lIneage Tracking in Spatial omics), a comprehensive and adaptable computational framework for analyzing spatially resolved BCR sequencing data. 
</p>

<p align="center">
  <img src="https://github.com/zenglab-pku/BAITS/raw/master/docs/_static/BAITS_framework.png" width="500px">
</p>


## Introduction
<p>
BAITS comprises three core modules:
</p>
- ** Spatial Transcriptomics (ST) module **: identify B lymphocyte aggregates based solely on spatial transcriptomic data
- ** Immune Repertoire (IR) module **: quantify clonal expansion, clonal degree centrality, and other repertoire features using spatial BCR sequencing data
- ** SR module **: reveal patterns of clonal migration, expansion, and niche restriction by integrating spatial transcriptomic and BCR data

## Installation
1. Create a conda or pyenv environment and install Python >= 3.10,<3.13 
```bash
conda create --name BAITS python=3.10
```
2. Pip install BAITS
```bash
conda activate BAITS
pip install BAITS
```

This example is based on a Linux CentOS 7 system 

## Contribution

If you found a bug or you want to propose a new feature, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/zenglab-pku/BAITS/issues
