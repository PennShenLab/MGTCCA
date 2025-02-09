## Multi-Group Tensor Canonical Correlation Analysis

This repository holds the official code for the paper [Multi-Group Tensor Canonical Correlation Analysis](https://dl.acm.org/doi/abs/10.1145/3584371.3612962).

### 🎯 Abstract

Tensor Canonical Correlation Analysis (TCCA) is a commonly employed statistical method utilized to examine linear associations between two sets of tensor datasets. However, the existing TCCA models fail to adequately address the heterogeneity present in real-world tensor data, such as brain imaging data collected from diverse groups characterized by factors like sex and race. Consequently, these models may yield biased outcomes. In order to surmount this constraint, we propose a novel approach called Multi-Group TCCA (MG-TCCA), which enables the joint analysis of multiple subgroups. By incorporating a dual sparsity structure and a block coordinate ascent algorithm, our MG-TCCA method effectively addresses heterogeneity and leverages information across different groups to identify consistent signals. This novel approach facilitates the quantification of shared and individual structures, reduces data dimensionality, and enables visual exploration. To empirically validate our approach, we conduct a study focused on investigating correlations between two brain positron emission tomography (PET) modalities (AV-45 and FDG) within an Alzheimer's disease (AD) cohort. Our results demonstrate that MG-TCCA surpasses traditional TCCA in identifying sex-specific cross-modality imaging correlations. This heightened performance of MG-TCCA provides valuable insights for the characterization of multimodal imaging biomarkers in AD.

### 🔨 Usage

The method, [MGTCCA](https://github.com/PennShenLab/MGTCCA/blob/main/gtcca.m), is introduced in this work.
