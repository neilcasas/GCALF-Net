**UNIVERSITY OF SANTO TOMAS COLLEGE OF INFORMATION AND COMPUTING SCIENCES DEPARTMENT OF COMPUTER SCIENCE** 

**GCALF-Net: Modified PDHD-Net with Adaptive Frequency Filtering and Cross-Attention Fusion for Gleason Grade Group Classification using bpMRI** 

A Thesis Proposal Presented to the Department of Computer Science College of Information and Computing Sciences University of Santo Tomas 

In Partial Fulfilment of the Requirements for the Degree Bachelor of Science in Computer Science 

By 

**Bobiles, Edmond John D.R. Casas, Neil Alfonz C. Padua, Sydney Alison P.** 

Adviser: 

**Acula, Donata D.** 

**April 2025** (Ths 1) 

1 

## **Table of Contents** 

**Chapter I    The Problem and Its Background................................................................2** A. Introduction...............................................................................................................2 B. Background of the Study...........................................................................................4 C. Theoretical Framework............................................................................................. 5 D. Conceptual Framework.............................................................................................9 E. Statement of the Problem........................................................................................ 14 F. Objectives................................................................................................................ 15 G. Scope and Limitations.............................................................................................16 H. Significance of the Study........................................................................................17 I. Definition of Terms.................................................................................................. 19 **Chapter II    Review of Related Literature and Studies...............................................22** A. Challenges in Gleason Grading for Prostate Cancer...............................................23 B. Convolutional Neural Networks in Prostate Cancer Imaging.................................25 C. Swin Transformers in Prostate Cancer Imaging......................................................28 D. Hybrid CNN-Swin Architecture in Prostate Cancer Imaging.................................30 E. Frequency Domain Feature Enhancement in MRI Modalities................................32 F. Cross-Attention Mechanisms for Multi-Branch Feature Fusion............................. 34 G. GradCam & Explainability in Medical Artificial Intelligence................................37 H. Synthesis................................................................................................................. 38 **Chapter III Research Design and Methodology........................................................... 43** 

2 

A. Hypothesis...............................................................................................................43 B. Research Methods................................................................................................... 45 C. Research Design......................................................................................................47 E. Sampling and Data Gathering Procedure................................................................ 55 F. Statistical Treatment of the Data..............................................................................58 **References.........................................................................................................................71** 

3 

## **List of Figures** 

|Figure|1.1:|Framework of PDHD-Net (Wang et al., 2025)|8|
|---|---|---|---|
|Figure|1.2:|Framework of the Proposed GCALF-Net|12|
|Figure|2.1:|Convolutional Neural Network (Lecun et. al., 1998)|27|
|Figure|2.2:|Poisson Ordinal Network for Gleason group estimation (Xu et. al.,||
|2024)|||29|
|Figure|2.3:|SwinBTS Architecture (Jiang et al., 2022)|31|
|Figure|2.4:|The PDHD-Net dual-branch architecture (Wang et. al, 2025)|33|
|Figure|2.5:|Frequency Selection Segmentation Network (Cai et al., 2025)|35|
|Figure|2.6:|Attention mechanism from IFC-Net (Cai et al., 2025)|37|
|Figure|3.1.|Research Design Diagram|54|



4 

## **List of Tables** 

|Table|2.1:|Synthesis Table|42|
|---|---|---|---|
|Table|3.1:|Characteristic of PI-CAI Grand Challenge Dataset|58|
|Table|3.2.|5x5 Confusion Matrix|67|
|Table|3.3:|Five-point Likert Scale|72|



5 

## **Chapter I    The Problem and Its Background** 

This chapter presents the introduction, the background of the study, theoretical framework, conceptual framework, statement of the problem, scope and limitations, significance of the study, and definition of terms. 

## **A. Introduction** 

Prostate cancer (PCa) is a type of cancer marked by uncontrolled division of cells in the prostate gland leading to unusual growth of the prostate gland (Schatten, 2018). The prostate is a small gland found in the male reproductive system that helps in making the fluid part of the semen (Singh & Bolla, 2023). This cancer is found in men worldwide and is the second most common diagnosis overall. Prostate cancer ranks as the third most common cancer among Filipino men, with 9,764 new cases and 3,850 deaths recorded in 2022, reflecting an age-standardized incidence rate of 24.1 per 100,000 (Ko et al., 2025; MIMS Oncology Honorary Editorial Advisory Board, 2025). 

Globally, it is the second most common cancer with more than 1,460,000 estimated cases and around 400,000 deaths in 2022, with the United States leading the highest percentage in the world (Bray et al. 2024). The growing impact of prostate cancer is mainly due to aging populations and better detection methods, especially in places like North America and Europe. In these regions, regular screening has led to higher reported case numbers (Chu et al., 2025; Zhao et al., 2025). However, mortality rates in cancer remain much higher in low- and middle-income countries, due to limited access to early detection tools and treatments (Stefan & Tang, 2023). The global differences highlight 

6 

the importance of diagnosing prostate cancer early and accurately, in order for the patients to get the right treatment and have better outcomes. 

Central to evaluating the severity of prostate cancer, the Gleason Pattern (GP) score is the most effective grading system where the tissue is studied to determine the abnormality of the prostate cancer cell from 1 to 5 (Gordetsky & Epstein, 2016). The Gleason Score (GS) that has been derived from this grading is significant not only in determining the stage of the disease, but also in guiding appropriate treatment, since higher scores indicate more aggressive forms of cancer that may require more immediate or intensive treatment (Short, Warren, & Varma, 2019). 

Currently, histopathological biopsy is the gold standard for determining the GS. However, given that PCa is often a slower growing cancer as compared to other cancers, there is generally less clinical urgency to immediately attempt an invasive procedure (Fiorentino et al., 2020). As a result, the use of multiparametric magnetic resonance imaging (mpMRI) became the preferred non-invasive alternative in the detection and evaluation of prostate cancer (Patel et al., 2019). In addition to being non-invasive, studies have shown that MRI-first strategies are cost-effective compared to standard biopsy pathways. By reducing unnecessary biopsies by 20-50%, they help lower the overall procedural burden and related histopathological costs, even when considering the higher upfront cost of imaging (Venderink et al., 2017; Yun et al., 2023). Despite this, the use of mpMRI has its limitations and alone cannot definitely grade tumors, which suggests that there is a need for more advanced methods that can connect non-invasive imaging with more accurate cancer assessment. 

7 

Although deep learning models have demonstrated promising results in analyzing MRI scans for prostate cancer assessment, most of the existing methods are still limited without resorting to invasive procedures. This study proposes a hybrid deep learning architecture based on Convolutional Network (CNN) and Swin Transformer to classify the aggressiveness of prostate cancer using biparametric magnetic resonance imaging (bpMRI), without relying on invasive biopsy. Through the use of bpMRI as the input, the proposed approach supports a more accessible diagnostic pathway, especially in clinical settings where biopsy resources are limited. 

## **B. Background of the Study** 

Clinical diagnosis for PCa typically begins with a Prostate-Specific Antigen (PSA) test, followed by an MRI scan for positive cases. Radiologists score these images using the Prostate Imaging Reporting and Data System (PI-RADS); where only lesions with PI-RADS  3 are candidates for an invasive biopsy (Werner, et al. 2023). ≥ 

However, from the PI-RADS 3 lesions only 26.8% was found to have overall PCa and 14.6% have clinically significant after a biopsy test (Schlenker, et al. 2019). This low accuracy has led medical professionals to seek supplementary diagnostic factors before patients are committed to a biopsy (Nikola & Bittencourt, 2023). 

When a biopsy is performed, malignancy is categorized via the Gleason Grade Group (GGG) system or ISUP System (Leenders et al. 2020). This system simplifies the Gleason Score into five distinct risk categories: 

- GGG1 (ISUP 1): Low-grade cancer, defined by GS (3 + 3) or less 

- GGG2 (ISUP 2): Intermediate Risk with Favorable Prognosis, defined by GS (3 + 4 ) 

8 

- GGG3 (ISUP 3): Intermediate Risk with Unfavorable Prognosis, defined by GS (4 + 3 ) 

- GGG4 (ISUP 4): High-Risk, defined by GS (4 + 4, 3 + 5, 5 + 3) 

- GGG5 (ISUP 5): Extremely High-Risk, defined by GS (4 + 4, 3 + 5, 5 + 3) 

Because PCa is often slow-progressing, frequent biopsies within the first few years of detection can be unnecessarily invasive (NIH, 2022). In line with this, an MRI-first diagnosis has become increasingly common today. The adoption of MRI first can eliminate 57 percent of overdiagnoses related to clinically insignificant cancer while maintaining a low risk of missing incurable disease (Hugosson et al., 2024). 

To improve PCa prognosis prediction, this study builds upon the Prostate Dual-branch Hybrid Domain Prediction Network (PDHD-Net) proposed by Wang et al. (2025), which successfully integrated CNN and Swin Transformer with frequency-domain and multi-scale feature learning for GGG classification. However, limitations remain in its fixed frequency decomposition strategy, indirect feature fusion mechanism, and reliance on multiparametric MRI inputs. To address these issues, the present study adapts the PDHD-Net framework by introducing a learnable frequency representation, a cross-attention-based fusion mechanism, and a Gradient-weighted Class Activation Mapping (Grad-CAM)  component to improve adaptability, clinical feasibility, and model explainability. 

## **C. Theoretical Framework** 

This study is based on the architectural framework of Wang et al. (2025), specifically their PDHD-Net, which demonstrated that a parallel CNN and Swin 

9 

Transformer hybrid architecture with the addition of frequency domain decomposition 

achieves superior Gleason Grade Group prediction by utilizing mpMRI. 

**Figure 1.1:** _Framework of PDHD-Net (Wang et al., 2025)_ 

As illustrated in Figure 1.1, the theoretical foundation of this study is built upon the three core components of PDHD-Net: the Frequency-Band-Shunting Feature Enhancement module (FDSF), the ConvSwin3D dual-branch encoder, and the 3D Learnable Bidirectional Feature Pyramid Network (3D-LDFPN). Additionally, when it comes to input, Wang et al. (2025) used four MRI sequences: T2-weighted imaging (T2W), Apparent Diffusion Coefficient maps (ADC), and two Diffusion-Weighted Imaging sequences at b-values of 1000 and 3000 (DWI1000 and DWI3000), combined into a four-channel volumetric input. 

10 

The first component, the FDSF module, establishes the theoretical basis for frequency-domain feature separation. Wang et al. (2025) applied a 3D Fast Fourier Transform (FFT) to decompose the MRI volume into low-frequency components, which capture overall organ structure and global patterns, and high-frequency components, which emphasize lesion edges and local texture details. These separated components are then routed to the Swin Transformer and CNN branches respectively. In the PDHD-Net, Frequency Manipulation Block (FMB) performs the separation using a Fixed Decomposition Representation (FDR), which is implemented as a spherical mask centered in the frequency spectrum. A manually defined hyperparameter α then controls this mask, which determines the radius of the sphere, which effectively sets a fixed cutoff between low and high frequencies. The same α value is applied uniformly to every input MRI volume, similar to other fixed frequency decomposition approaches (Ewaidat et al., 2024; Li et al., 2020). This assumes that the optimal frequency boundary is identical across all cases, which is rarely true in clinical practice, as variability in patient anatomy, tumor characteristics, and imaging conditions often requires adaptive frequency selection (Luo et al., 2024; Rao et al., 2021). 

The second component, ConvSwin3D, establishes the theoretical basis for parallel local and global feature extraction in prostate MRI analysis. According to Wang et al. (2025), using only a CNN or a Transformer encoder is not sufficient for accurate Gleason grading due to the simultaneous presence of fine local texture variations and broader anatomical structures. To address this, ConvSwin3D processes the same input through two parallel branches. The CNN branch extracts local features using a series of 3D convolutional layers that progressively capture fine-grained spatial patterns within the 

11 

prostate volume. The Swin Transformer branch captures long-range dependencies and contextual relationships by applying self-attention mechanisms across the volumetric space. In the PDHD-Net, the outputs of the two branches are fused using the Window-based self-attention fusion (WAF) module, where features are concatenated and processed through window-based self-attention. While this allows feature integration, such fusion remains symmetric and implicit, which lacks explicit cross-branch guidance. This limitation is also noted in other hybrid architectures which rely on concatenation-based self-attention (Ding et al, 2024; Cai et al., 2025).  Wang et al. (2025) reported reduced sensitivity at the boundary between GGG 2 and GGG 3 (Gleason 3+4 vs 4+3), indicating that this fusion approach struggles in clinically critical fine-grained classification tasks, where more targeted feature interaction is required (Badar et al., 2025). 

The third component, the 3D-LDFPN, provides the theoretical basis for multi-scale feature integration in volumetric prostate cancer grading. Wang et al. (2025) explained that traditional feature pyramid networks with single-direction information flow are insufficient for capturing lesions at varying scales. To address this, PDHD-Net utilizes 3D-LDFPN which is composed of a bidirectional structure: a top-down path that propagates semantic information from low-resolution features to higher-resolution layers, and a bottom-up path that reinforces spatial detail from high-resolution features into deeper layers. At each level, multi-scale features are fused using learnable weights that are updated during training, making the fusion process more flexible compared to fixed-weight approaches. 

12 

Wang et al. (2025) reported an averaged sensitivity of 85.5% for GGG 1–4 compared to 96.7% for GGG 5. This poses a clinical challenge, as accurate differentiation between GGG 2 and GGG 3 is critical. Specifically, Gleason score 4+3 (GGG 3) indicates a worse prognosis than 3+4 (GGG 2), which represents a key clinical threshold where management typically shifts from active surveillance to active treatment. Despite the overall performance of PDHD-Net, this boundary remains difficult to classify due to its subtle morphological differences and clinical significance. 

As illustrated in Figure 1.1, the present study adopts PDHD-net as its theoretical basis while introducing two targeted modifications to address these limitations. Specifically, the FDR within the FMB of the FDSF module, and the WAF within ConvSwin3D. In addition, the original four-channel MRI input is replaced with a three-channel bpMRI setup, and clinical interpretability is incorporated through Grad-CAM visualization. The remaining components — the CNN branch, Swin Transformer branch, and 3D-LDFPN decoder — are retained without modification. 

## **D. Conceptual Framework** 

This study proposes the Gleason Grading Cross-Attention Learnable Frequency Network (GCALF-Net), conceptually derived from the PDHD-Net proposed by Wang et al. (2025), which serves as the foundational architecture for the development of the model. Building upon this framework, GCALF-Net introduces targeted modifications to address limitations in frequency decomposition and feature fusion, while also adapting the model for bpMRI input instead of the original mpMRI input used in the PDHD-Net. 

13 

**Figure 1.2:** _Framework of the Proposed GCALF-Net_ 

As shown in Figure 1.2, the GCALF-Net retains the overall architectural structure of PDHD-Net while introducing Learnable Frequency Filter (LFF) in place of the fixed FDR and Cross-Attention Fusion (CAF) in place of WAF. The name GCALF-Net reflects its key contributions: Gleason Grading as the target task, Learnable Frequency as the adaptive frequency weighting mechanism, and Cross-Attention as the modified feature fusion mechanism. The study retains the three core components from Wang et al. (2025), namely FDSF, ConvSwin3D, and 3D-LDFPN, with modifications applied only to specific mechanisms within FDSF and ConvSwin3D. Additionally, the model is adapted 

14 

to a bpMRI input setting instead of the original multiparametric MRI mpMRI configuration used in PDHD-Net. 

The primary input of the GCALF-Net is a 3D bpMRI volume obtained from the PI-CAI (Prostate Imaging: Cancer AI) public dataset, consisting of cases with GGG labels based on biopsy-confirmed ISUP grading. Each input includes three MRI sequences: T2W, ADC maps, and DWI, forming a three-channel volumetric input that captures complementary characteristics of prostate tissue. T2W provides anatomical structure and zonal boundaries, ADC reflects diffusion restriction associated with cellular density, and high b-value DWI highlights regions of restricted diffusion linked to tumor aggressiveness. This three-channel setup is different from input utilized in PDHD-net, which comprises four MRI sequences (T2W, ADC, DWI1000, and DWI3000). In clinical settings, high b-value DWI sequences such as DWI3000 are not always available, making the proposed bpMRI setup more practical for potential clinical application while retaining essential diagnostic information. 

The main process involves a modified hybrid CNN–Swin Transformer architecture adapted from Wang et al. (2025), with two targeted architectural modifications. The overall architecture, including the CNN branch, Swin Transformer branch, and 3D-LDFPN decoder, is retained from Wang et al. (2025). Only the frequency decomposition mechanism in the FDSF module and the feature fusion mechanism in the ConvSwin3D encoder are modified, as these were identified as the primary sources of limitation in the original framework. 

The first modification addresses the fixed frequency decomposition in the FDSF module. In the original framework, Wang et al. (2025) use a fixed spherical mask 

15 

controlled by a manually defined hyperparameter α to separate the input MRI volume into low- and high-frequency components. This applies the same frequency boundary to all inputs regardless of patient-specific variation, resulting in the model relying on a predefined assumption about how frequency information should be divided. Rao et al. (2021) showed that fixed frequency operations can limit model performance because they impose human-defined assumptions that may not reflect the true frequency distribution of the data, and demonstrated that learnable global filters optimized directly during training produce more effective feature representations. This principle has also been applied in medical imaging such as the study of Tragakis et al. (2024), in which, through the GLFNet architecture, used LFF for medical image segmentation and reported strong performance across multiple datasets. To address this, the present study replaces the fixed spherical mask with a LFF implemented as a trainable weight matrix W applied through element-wise multiplication in the Fourier domain, following the approach of Rao et al. (2021). During training, W is updated through backpropagation, allowing the model to adjust how frequency information is weighted based on the input data rather than relying on a fixed separation rule. 

The second modification addresses the feature fusion mechanism in the ConvSwin3D encoder. In Wang et al. (2025), CNN and Swin Transformer features are concatenated and processed using window-based self-attention. While this enables integration, the interaction is symmetric and implicit which does not allow one branch to directly and selectively query the other. This limitation is most significant at the boundary between GGG 2 and GGG 3, corresponding to Gleason 3+4 and Gleason 4+3, where Wang et al. (2025) reported lower sensitivity values, indicating difficulty in 

16 

distinguishing clinically important and visually subtle cases. Chen et al. (2022) pointed out that simply combining convolutional and attention-based features through sequential operations or applying self-attention to concatenated representations may not lead to effective interaction between local and global features. Meanwhile, Cai et al. (2025), using the Interactive CNN and Transformer for Cross-Attention Fusion Network (IFC-Net) architecture, demonstrated that cross-attention fusion allows directed and asymmetric information exchange between CNN and Transformer components, better reflecting their complementary roles and leading to improved classification performance. Similarly, Badar et al. (2025), through Multi-Scale Cross and Self-Attention Network (MSCAS-Net), showed that combining self-attention and cross-attention mechanisms across multiple scales enhances the detection of subtle features, enabling more accurate discrimination between visually similar classes. To address this, the present study replaces WAF with CAF, enabling directed, branch-conditioned interaction between the CNN and Swin Transformer branches. 

The 3D-LDFPN component of PDHD-Net is retained in GCALF-Net. The model also produces the same outputs in PDHD-Net.  The first is a detection result that localizes suspicious lesions within the prostate MRI volume, evaluated using FROC analysis at specified false-positive thresholds. The second is a five-class risk prediction generated for each detected lesion. The third is a voxel-level segmentation mask that delineates the boundary of the detected lesion, evaluated using the Dice Similarity Coefficient (DSC) to measure spatial overlap with ground-truth annotations. Additionally, Grad-CAM heatmaps are generated after prediction by extracting gradients from the final convolutional layer of the classification branch. 

17 

Overall, the inputs, processes, and outputs form a complete pipeline for non-invasive GGG prediction of prostate cancer using bpMRI. The proposed modifications address the limitations of fixed frequency decomposition and indirect feature fusion identified in Wang et al. (2025). As a result, the model becomes more adaptive in feature processing, more precise in classification, and more interpretable in a clinical context. 

## **E. Statement of the Problem** 

The general problem of the study is reliance on invasive procedures to determine prostate cancer aggressiveness through Gleason score determination via tissue biopsy. This is due to the inability of Magnetic Resonance Imaging (MRI) to classify prostate tumor grades by itself. While current models have shown promising results using deep learning techniques seen in the paper of Wang et al. (2025), there is a performance disparity across lower and intermediate grade groups. Wang et al. (2025) reported an averaged sensitivity of 85.5% for GGG 1-4 compared to a 96.7% for GGG 5. This poses an issue as it is critical to differentiate between GGG 2 from GGG 3, as the GS (4 + 3) in GGG 3 is a worse prognosis than the (3 + 4) in GGG 2, where it is a major clinical threshold in transitioning from monitoring of the cancer to active medical treatment. The classification task is a five-class problem: GGG 1 (low risk), GGG 2 (favorable intermediate risk), GGG 3 (unfavorable intermediate risk), GGG 4 (high risk), and GGG5 (extremely high risk).  To address this, the model in this research implements a modified PDHD-Net architecture with a learnable frequency filter and cross-attention fusion. 

## **Specifically, this study aims to answer the following:** 

_1. How does the adapted PDHD-Net baseline perform on the PI-CAI dataset ?_ 

18 

_**2.** How do the proposed architectural modifications perform in five-class Gleason_ 

_Grade Group classification? Specifically, how does performance vary under the following configurations:_ 

   - a) PDHD-Net with Learnable Frequency Filter (LFF) only 

   - b) PDHD-Net with Cross-Attention Fusion (CAF) only 

   - c) full proposed model (GCALF-Net: LFF + CAF) 

_**3.** Is there a significant improvement in GCALF-Net compared to the baseline_ 

_PDHD-Net?_ 

_**4.** Are the Grad-CAM-generated heatmaps acceptable based on PI-RADS version 2_ 

_clinical criteria as evaluated by three (3) Urologists?_ 

## **F. Objectives** 

The primary objective of this study is to develop and evaluate a modified hybrid deep learning architecture, specifically the PDHD-Net proposed by Wang et al. (2025), that integrates CNNs and Swin Transformers to enable accurate, non-invasive classification of PCa aggressiveness using 3D bpMRI sequence of TW2, ADC, and DWI from the PI-CAI public dataset. This addresses the current limitations of bpMRI in independently determining tumor aggressiveness as measured by the Gleason Score. Although existing deep learning approaches by Wang et al. (2025) have demonstrated promising classification performance, this study aims to address identified limitations through architectural modifications. Thus, the specific objectives of this study are as follows: 

19 

1. To establish and evaluate the baseline performance of the adapted PDHD-Net architecture on the PI-CAI dataset across the five-class Gleason Grade Group classification framework (GGG 1, GGG 2, GGG 3, GGG 4, and GGG 5). 

2. To evaluate the classification performance of the three model configurations: PDHD-Net with the Learnable Frequency Filter (LFF) only, PDHD-Net with the Cross-Attention Fusion (CAF) only, and the full proposed GCALF-Net incorporating both — in order to determine the individual and combined contribution of each architectural modification to five-class Gleason Grade Group classification. 

3. To determine whether the full proposed model (GCALF-Net) yields a statistically significant improvement in classification performance compared to the adapted PDHD-Net model. 

4. To assess the clinical interpretability of the proposed model through Grad-CAM heatmap visualizations, evaluated against PI-RADS version 2 clinical criteria by three (3) urologists. 

## **G. Scope and Limitations** 

This study focuses on the development of a hybrid deep learning architecture that incorporates a Convolutional Neural Network (CNN) and a Swin Transformer for the non-invasive classification of prostate cancer aggressiveness using T2-weighted (T2W), Apparent Diffusion Coefficient (ADC), and Diffusion-Weighted Imaging (DWI) scans derived from biparametric MRI (bpMRI). The classification is based on five Gleason Grade Groups (GGG): low risk (GGG 1), favorable intermediate risk (GGG 2), 

20 

unfavorable intermediate risk (GGG 3), high risk (GGG 4), and extremely high risk (GGG 5). However, the study is limited to the following: 

- The study exclusively utilizes the PI-CAI Challenge dataset, which, while multi-center and multi-vendor (11 institutions, Siemens and Philips), remains a single publicly available dataset. 

- The dataset does not include real-time clinical data, and model performance in actual clinical practice has not been evaluated. 

- The study does not aim to replace histopathological biopsy but rather to support physicians' clinical judgment. 

- The hybrid architecture is limited to bpMRI sequences (T2W, ADC, DWI) and does not incorporate full multiparametric MRI (mpMRI) sequences such as dynamic contrast-enhanced (DCE) imaging. 

- Demographic and clinical patient variables (e.g., PSA levels and age) were not included in the classification model. 

## **H. Significance of the Study** 

This study addresses a significant diagnostic gap in prostate cancer management. Accurate assessment of tumor aggressiveness using the Gleason Score is essential for informing treatment decisions; however, this process currently relies on invasive histopathological biopsy. Although biparametric MRI is a preferred non-invasive approach for prostate cancer detection and assessment, it cannot reliably grade tumors, highlighting the need for advanced computational methods that combine non-invasive imaging with precise cancer evaluation. This study advances non-invasive, automated 

21 

tumor grading as a complement or potential alternative to biopsy in appropriate clinical contexts. Existing deep learning models face persistent challenges in accurately classifying Gleason Grade Groups, as morphological similarities among adjacent MRI grades hinder precise stratification. The architectural improvements introduced in this study address these classification difficulties, thereby increasing the clinical value of non-invasive Gleason grading. This research provides immediate benefits to relevant stakeholders: 

- **Prostate Cancer Patients.** The tool offers a highly accurate, non-invasive diagnostic solution for Gleason Grade Group classification, reducing dependence on invasive biopsy and allowing easier determination of tumor aggressiveness, especially in environments with limited biopsy resources. 

- **Urologists and Radiologists.** The tool delivers an interpretable, AI-driven decision-support system that categorizes tumor aggressiveness across well-defined grade boundaries, enabling more confident treatment decisions rooted in non-invasive bpMRI imaging. 

- **Pathologists.** The tool reduces repetitive workload from manual microscopic grading. By expediting initial aggressiveness stratification, it lessens diagnostic fatigue and allows pathologists to focus on complex or ambiguous cases. 

- **Research Institutions.** The findings of this study can serve as a foundation for further research on deep learning applications for non-invasive imaging in prostate cancer diagnosis. This can lead to more advanced models for Gleason Grade Group classification and other oncological and urological conditions. Ultimately, this research can improve the accuracy, accessibility, and 

22 

cost-effectiveness of prostate cancer assessment in clinical and diagnostic settings. 

- **Healthcare Institutions.** Once GCALF-Net has been trained, it would only require inference level during deployment, which is significantly less resource-intensive than training and does not require per-patient retraining. Hospitals that already incorporate bpMRI infrastructure can integrate the system without additional licensing or cloud computing costs, making it practical and cost-effective addition to diagnostic existing diagnostic workflows, even in resource-limited settings. 

- **Computer Science Field:** Applying the proposed GCALF-Net architecture to classify prostate cancer aggressiveness using bpMRI can provide valuable insights into the performance of hybrid CNN–Swin Transformer models. Information about the model's adaptive frequency filtering and cross-attention fusion mechanisms may be used by other researchers for medical image classification tasks beyond prostate cancer grading. 

## **I. Definition of Terms** 

To ensure clarity and consistency, the following key terms used in this study are defined as follows: 

## **Area Under the Receiver Operating Characteristic Curve (AUROC).** 

AUROC measures how accurately the model ranks positive cases above negative cases across all thresholds. In this study, AUROC is used to evaluate how well the model distinguishes between adjacent Gleason Grade Groups. 

23 

**Apparent Diffusion Coefficient (ADC).** Values derived from Diffusion-Weighted Imaging (DWI) on an mpMRI as an imaging biomarker to detect and characterize Prostate Cancer. 

**Biparametric Magnetic Resonance Imaging (bpMRI).** It is a faster and more cost-effective, yet equally accurate alternative to mpMRI. The PI-CAI dataset used in this study consists of bpMRI sequences including T2W, DWI, and ADC. 

**Convolutional Neural Network (CNN).** A deep learning architecture highly effective for visual data processing that utilizes convolutional layers to extract pertinent local feature maps from input images. 

**Cross-Attention Fusion (CAF)** . An interactive feature integration module that enables direct, bidirectional information exchange between Convolutional Neural 

Network (CNN) and Transformer branches, allowing one network stream to selectively query and retrieve relevant features from the other. 

**Diffused-Weighted Imaging (DWI).** An MRI technique used to locate Prostate Cancer cells by measuring the movement of water molecules within tissue. 

## **Free-response Receiver Operating Characteristic (FROC) Analysis.** It is an 

evaluation metric used to assess performance of detection in computer vision and medical imaging. 

## **Gradient-weighted Class Activation Mapping (Grad-CAM).** A visualization 

technique used to provide clinical explainability for deep learning models by generating heatmaps that highlight the specific regions of an image most influential to the model's decision-making process. 

24 

**Gleason Score (GS)** .  A numerical score assigned by a pathologist based on the microscopic appearance of prostate tissue from a biopsy, used to evaluate the tumor's aggressiveness. 

**Gleason Grade Group (GGG).** A five-tiered, standard method used to grade prostate and give a more accurate prognosis than the traditional Gleason Score. 

**Hematoxylin and Eosin (H&E).** A widely used staining technique in 

histopathology where hematoxylin stains cell nuclei blue and eosin stains the surrounding tissue pink, allowing pathologists to examine tissue structure under a microscope. 

**Histopathology.** The microscopic examination of tissue samples to study the appearance and structure of cells, used in medicine to diagnose diseases such as cancer. 

**ISUP Grade Group.** A grading system developed to provide a more accurate and easier to understand measure of how quickly prostate cancer may spread, determined using prostate biopsy samples. It uses a scale of 1 to 5 where higher grade groups indicate a greater risk of the cancer being aggressive and spreading rapidly. 

**Learnable Frequency Filter (LFF)** . A dynamic filtering mechanism applied in the Fourier domain that utilizes a trainable weight matrix to adaptively separate and 

emphasize spatial frequency components based on input data, replacing fixed-parameter decomposition methods. 

**Magnetic Resonance Imaging (MRI).** A non-invasive medical imaging technique that uses a powerful magnetic field and computer-generated radio waves to create detailed, high-resolution images of the organs and tissues within the body. 

**Multi-class Classification.** A type of machine learning task where the model categorizes input data into one of three or more distinct classes. 

25 

**Multiparametric MRI (mpMRI).** A special type of MRI scan that produces a more detailed picture of the prostate gland than a standard MRI scan does. 

**PI-RADS (Prostate Imaging Reporting and Data System).** A structured 

reporting framework used in multiparametric prostate MRI to evaluate suspected prostate cancer in treatment-naive prostate glands. 

**Prostate Cancer (PCa)** .  A type of malignancy that develops in the prostate gland of the male reproductive system, often progressing slowly but capable of spreading to other parts of the body. 

**Prostate-Specific Antigen (PSA)** . A protein produced by the prostate gland whose blood levels are used to screen for prostate cancer and monitor prostate health. 

**Receiver Operating Characteristic (ROC) Analysis.** A graphical tool used to evaluate the performance of binary classification. 

**Swin Transformer** .  A hierarchical Vision Transformer that utilizes a shifted window-based multi-head self-attention mechanism, dividing inputs into local windows to capture global contextual dependencies while maintaining linear computational complexity, making it efficient for processing high-resolution medical images. 

**T2-Weighted (T2W) MRI.** A specific type of magnetic resonance imaging sequence that highlights the anatomical structure of the prostate gland and is primarily utilized by radiologists to detect the presence of tumors. 

**Transformer** . A deep learning architecture that relies on a self-attention mechanism to weigh the significance of different parts of input data, enabling highly efficient parallel processing of sequential information like text. 

26 

**Vision Transformer** . A model that adapts the Transformer architecture for 

computer vision by breaking images into a sequence of fixed-size patches and processing them as "tokens," similar to how words are treated in natural language processing. 

27 

## **Chapter II    Review of Related Literature and Studies** 

This chapter contains an in-depth analysis of studies closely related to this research study's chosen topic. The literature review on this chapter includes concepts, theories, algorithms, accuracy, complexity, definitions, and applications applied in the studies. Lastly, a synthesis table comparing various methods used in medical neuroimaging tools is presented in the last part of this chapter. 

## **A. Challenges in Gleason Grading for Prostate Cancer** 

Gleason grading, first introduced by Dr. Donald Gleason, assesses the aggressiveness of prostate cancer by microscopic examination of prostate tissue and informs optimal management strategies (Gleason, 1997). In the conventional process, urologists or surgeons obtain tissue via biopsy or surgery, after which pathologists apply standard histopathological techniques and stain specimens with hematoxylin and eosin (H&E) prior to microscopic evaluation (Bansal et al., 2025). However, the traditional Gleason grading method faces significant challenges. Notably, inter-observer variability arises when different pathologists assign varying Gleason scores to the same tissue slide, and subjectivity in pattern recognition impedes consistent scoring (Eble et al., 2004; Epstein et al., 2005). Intraobserver variability is also evident; for example, when urologists graded 81 slides and re-examined 47 slides, the agreement rate was only 77%, indicating bias even within the same observer (Melia et al., 2006). 

Furthermore, Gleason grading depends on manual microscopic evaluation by pathologists, which not only makes it vulnerable to interobserver variability and consistency but may also contribute to diagnostic fatigue due to its repetitive, detail-intensive nature. Khatab et al. (2023) stated that manual diagnostic tasks in 

28 

pathology contribute to heavy workloads and are linked to burnout and exhaustion among pathologists, highlighting the labor-intensive demands of tasks such as Gleason grading. 

These inter- and intra-observer discrepancies in histopathological grading could lead to the common issue of under- and overgrading of the GS score, particularly with GS patterns 3 and 4. Cribriform GS pattern 4 was underdiagnosed as GS pattern 3, and this poses a danger, as GS pattern 4 is an invasive growth mistaken for the benign GS pattern 3 (Ozkan et al., 2016). Lack of accuracy has been a common error for predicted GS patterns as it is subject to observer variation, with pathologists having different proficiencies in reading the GS scores, especially for needle-thin tissue biopsy samples (Klotz et al. _,_ 2009). 

Beyond reliability concerns, the Gleason grading process is inherently invasive, requiring tissue extraction that carries procedural risks. Grading also depends on human interpretation, which limits both scalability and accuracy. Postoperatively, 63% of patients experience complications such as hematuria, hematospermia, and rectal bleeding, while 2.29% face more severe complications, including sepsis in the early post-biopsy period (Borghesi et al., 2017). 

Moreover, in an effort to create a pain-averted method in PCa diagnoses, mpMRIs are increasingly being recommended since 2018 (Bratt et al., 2022), before a patient decides whether to perform a prostate biopsy, the gold standard technique in diagnosing and determining the aggressiveness of PCa. MRI-targeted biopsies have been shown to detect clinically significant PCas more effectively than the commonly used Transrectal-Ultrasound Scan (TRUS)-guided biopsy (Mate et al., 2023). Studies suggest that there remains a gap in non-invasive diagnostic methods for PCa, which mpMRIs fill, 

29 

with high sensitivity for diagnosis and helping avoid under- or overtreating PCa (Emekli et al., 2021). 

In line with this, innovations in assessing the Gleason score using mpMRI are being developed through deep learning and machine learning techniques. Currently, mpMRI readings are still read manually by pathologists, subject to human error, much like biopsy readings (Shen & Wu, 2023). By utilizing deep learning architectures like CNNs, subtle tissue patterns and spatial hierarchies that are too complex for humans to detect manually can be identified better. Furthermore, while the models provide improved accuracy, lesion localization, and segmentation through multimodal data, issues such as inconsistent data, small sample sizes, and image quality could create boundary overlap challenges when the image is being read by the AI (Prabhu et al. 2026). 

## **B. Convolutional Neural Networks in Prostate Cancer Imaging** 

**Figure 2.1:** _Convolutional Neural Network (Lecun et. al., 1998)_ 

Convolutional neural networks (CNNs) are among the most widely adopted deep 

learning architectures for automated prostate cancer detection and Gleason grading using 

magnetic resonance imaging (MRI). Originally introduced by LeCun et al. (1998), CNNs 

are a class of artificial neural networks that automatically learn spatially local features from raw input data by applying learnable convolutional filters, making them particularly 

30 

well-suited for image recognition and classification tasks. A principal advantage of CNNs is their capacity to learn hierarchical spatial features, such as textures, edges, and morphological patterns, directly from raw imaging data through stacked convolutional layers. This approach eliminates the need for manual feature engineering, as in earlier radiomics-based methods. De Vente et al. (2021) integrated ordinal coding with a 3D U-Net for simultaneous lesion detection and Gleason grade mapping in biparametric MRI. The transition from 2D to 3D CNN architectures has further expanded model capabilities, as volumetric networks can capture inter-slice spatial relationships that are lost when analyzing individual MRI slices. Zheng et al. (2024) demonstrated this advantage with WS-UNet, a weakly supervised 3D model that leveraged voxel-level spatial information to detect even MRI-invisible prostate cancers. Xu et al. (2024) combined Poisson ordinal regression with contrastive learning on 3D data to achieve sharper differentiation between adjacent Gleason grade groups. 

**Figure 2.2:** _Poisson Ordinal Network for Gleason group estimation (Xu et. al., 2024)_ 

Despite these strengths, CNN-based methods for Gleason grading exhibit several notable limitations. First, CNNs are inherently constrained by their local receptive fields. While convolutional filters are effective at capturing fine-grained local features, they 

31 

struggle to model long-range spatial dependencies across an entire MRI volume. This limitation is particularly problematic for understanding the global context of heterogeneous tumors. As Yang et al. (2026) observed, traditional 3D CNNs such as 3D ResNet are parameter-efficient but have difficulty capturing long-range correlations and performing multimodal fusion. U-Net-derived models also remain limited by the local nature of convolutional feature extraction. Second, most CNN-based studies have been restricted to binary classification, such as distinguishing clinically significant from non-significant cancer or high-grade from low-grade disease, rather than performing the full five-class Gleason Grade Group (GGG 1–5) stratification required for individualized 

treatment decisions. Liu et al. (2024) proposed a radiomics-CNN hybrid model that excelled at binary risk stratification, but its Sigmoid output and binary loss function design prevented it from capturing finer prognostic distinctions. Even studies that attempted multi-class schemes often merged GGG 4 and GGG 5 into a single high-risk category, as seen in Cao et al. (2019) with FocalNet, despite significant differences in invasiveness between these groups. Third, CNN models tend to exhibit high annotation dependence and poor cross-domain generalization, resulting in substantial performance degradation when applied to data from different institutions or MRI scanner manufacturers (Wang et al., 2025). 

In summary, CNNs have demonstrated high effectiveness in extracting local spatial features from prostate MRI and have achieved notable success in lesion detection, segmentation, and coarse-grained Gleason score prediction. Studies such as Duran et al. (2022) with ProstAttention-Net and Xinyu et al. (2023) with PFCA-Net have shown that incorporating attention mechanisms into CNN frameworks can partially address the 

32 

limited receptive field problem by guiding the network to focus on diagnostically relevant regions. However, the fundamental architectural constraints of CNNs, particularly their difficulty in capturing global contextual information and long-range dependencies, remain significant barriers to achieving the fine-grained, five-class Gleason grading accuracy required in clinical practice. These persistent limitations have motivated the exploration of transformer-based architectures, such as the Swin Transformer, which employ self-attention mechanisms to model global relationships across the entire input volume. Hybrid CNN-Transformer designs also aim to combine the local feature extraction strengths of CNNs with the global modeling capabilities of transformers. 

## **C. Swin Transformers in Prostate Cancer Imaging** 

As mentioned earlier, CNNs have difficulty in learning the global context for image analysis and information interaction. With this, Vision Transformers were used in the vision domain to address the weaknesses in the recognition of CNNs. Swin Transformers, in particular, is a hierarchical ViT aimed at being an effective image classifier, object detector, and semantic segmentator by implementing a _shifted-window_ -based self-attention approach where the image is divided in perfect-grid patches in windows into tokens, and are gradually merged with neighboring patches in deeper transformer layers. This produces a lower latency than the _sliding-window_ , but with the same modeling power (Liu et al. 2021). 

33 

**Figure 2.3:** _SwinBTS Architecture (Jiang et al., 2022)_ 

As illustrated in Figure 2.3, medical image targets are rarely perfect circles; often, they are scattered and span extensive areas. While CNNs are better at capturing the texture within lesion areas, they remain “blind” to the overall distribution of the organ, a limitation that Swin Transformers help overcome by learning accurate information about the spatial distribution of targets (Fu et al., 2024). Vision transformers, utilizing the multiple self-attention (MSA) module, divide an image into small patches for processing. This, however, has a tendency to produce border artifacts per patch; Swin solves this issue through its shifted window self-attention mechanism as well as its integration with CNNs (Liang et al., 2021). Additionally, while standard ViTs are powerful for global modeling, they suffer from memory limitations and high parameter counts when applied to 3D medical tasks. By restricting self-attention to shifted windows, Swin greatly reduces the number of parameters and computational complexity without compromising—and even improving—the feature learning capability for 3D volumetric images like MRIs, as demonstrated in the study by Jiang et al. (2022). 

34 

Moreover, Swin Transformers have versatility utility in multi-model data integration, such as MRI scans, across various oncological domains because of its ability to ex tract deep hierarchical representations from different kinds of scans and fuse them accurately (Yan et al. 2022). Again, this capability is anchored in the shifted-window mechanism in Swin. Specifically for prostate cancer, projecting heterogeneous data streams from a disparate sequence, such as T2W, ADC, and DWI scans, into a collective feature space using hierarchical Swin variants can identify global information lost in the traditional CNNs without the heavy computational cost of global attention (He et al. 2023). 

## **D. Hybrid CNN-Swin Architecture in Prostate Cancer Imaging** 

To address the limitations of CNNs and Swin Transformers discussed above, recent advancements have increasingly focused on hybrid architectures that synergize the two approaches. While CNNs excel at extracting localized, fine-grained structural details through their focused receptive fields, they struggle with broad contextualization. Conversely, Swin Transformers efficiently capture global semantic relationships through shifted-window self-attention but can occasionally miss subtle local textures. Fusing these architectures allows models to simultaneously evaluate minute local anomalies and overarching anatomical distributions, creating a robust framework for complex, fine-grained classification tasks. 

In the domain of prostate cancer imaging, this dual capability is crucial for automated Gleason grading. Identifying the aggressiveness of prostate cancer via multiparametric MRI (mp-MRI) requires the algorithm to detect subtle, micro-level textural differences in localized lesions while maintaining a global awareness of the 

35 

surrounding prostate anatomy. Hybrid frameworks that combine 3D CNNs with Vision Transformers have proven highly effective at bridging this gap, seamlessly processing complex 3D spatial domains to yield more precise, non-invasive Gleason score predictions. 

**Figure 2.4:** _The PDHD-Net dual-branch architecture (Wang et. al, 2025)_ 

Based on Figure 2.4, Wang et al. (2025) introduced the Prostate Dual-branch Hybrid Domain prediction Network (PDHD-Net) to achieve fine-grained stratification of the ISUP Gleason Grade Group (GGG 1-5). The PDHD-Net leverages a parallel dual-branch encoder that integrates a multi-stage 3D CNN with a Swin-Transformer, enhanced by an innovative frequency-band-shunting feature-enhancement strategy. This strategy directs low-frequency MRI components to the Swin-Transformer to capture global macro-structures, while high-frequency components are routed to the CNN to accentuate local lesion contours. However, despite achieving an exceptional 96.7% sensitivity for high-risk GGG=5 lesions, the PDHD-Net demonstrated a notable weakness 

36 

in accurately classifying lower Gleason grade groups, with sensitivities ranging from 83.3% to 87.3% for GGG 1 through 4. This highlights an ongoing clinical challenge in maintaining optimal feature extraction and detection accuracy across the entire spectrum of tumor aggressiveness. 

## **E.  Frequency Domain Feature Enhancement in MRI Modalities** 

Frequency domain analysis is a method of transforming images from spatial domain to the frequency domain and has become an important complement to traditional domain processing in medical image analysis (Mirabella, 2025). Unlike spatial domain methods that work directly on pixel intensities, frequency domain techniques, on the other hand, transform image data into a spectral representation. This technique is able to properly separate low-frequency components, which capture global structural and morphological information, from high-frequency components, which capture fine-grained edges and textures (Wang et al., 2025). This separation is particularly useful in prostate MRI, where lesions can look very similar to surrounding tissue in the spatial domain and by using frequency domain analysis, images are instead transformed into frequency domain for better medical image analysis. 

Early frequency domain approaches relied on fixed decomposition methods. Ewaidat et al. (2024) proposed the Frequency-Guided U-Net (GFNet), which integrates fast Fourier transform operations and attention filter gates into the encoder using predefined frequency band selection. Similarly, Li et al. (2020) developed Wavelet U-Net, replacing conventional pooling with discrete wavelet transform based on fixed decomposition levels. While these approaches demonstrated improvements over 

37 

spatial-only models, they apply uniform frequency processing across all inputs, limiting 

adaptability to variations in image characteristics and pathological features. 

**Figure 2.5:** _Frequency Selection Segmentation Network (Cai et al., 2025)_ 

As illustrated in Figure 2.5, recent studies have shifted toward more flexible and 

learnable frequency filtering approaches to address this limitation, such as the Frequency Selection Segmentation Network (FSSN) introduced by Luo et al. (2024). This network uses a trainable frequency band filter that can dynamically suppress less useful frequency components while highlighting those that are more relevant to the task. The filter weights are updated through backpropagation, allowing the model to learn which frequency ranges are most important based on the data. In a similar direction, Rao et al. (2021), through GFNet, showed that applying a learnable weight matrix in the Fourier domain enables adaptive emphasis of important frequency components, leading to better performance compared to fixed-mask methods. 

Wang et al. (2025) extended the use of frequency domain processing to prostate cancer grading by introducing a frequency-band-shunting strategy in PDHD-Net. 

38 

However, this approach still relies on a fixed spherical mask controlled by a predefined parameter α to separate low- and high-frequency components. Such a fixed separation does not account for differences across patients, MRI scanners, or imaging protocols, which may limit the model’s performance when applied to new datasets or to more commonly used biparametric MRI. 

Based on the learnable frequency filtering concepts presented by Rao et al. (2021) and Luo et al. (2024), the present study replaces the fixed frequency mask used by Wang et al. (2025) with a trainable weight matrix in the Fourier domain. This learnable frequency filter allows the model to adaptively emphasize the most relevant frequency components for the task, particularly for distinguishing between Gleason Grade Groups. This is especially important for the boundary between GGG 2 and GGG 3, where differences are more subtle and require more precise feature representation. 

## **F. Cross-Attention Mechanisms for Multi-Branch Feature Fusion** 

Feature fusion is important in hybrid deep learning architecture because it combines CNN and Transformer branches, because how well local and global features are integrated directly affects classification performance. Early fusion methods in hybrid encoders tended to rely on simple operations like concatenation or sequentially processing features from each branch. While these approaches allowed some level of combination, Ding et al. (2024) pointed out that merely stringing together convolutional and attention-based features, or applying self-attention to a concatenated representation, can still underutilize the complementary roles of local and global information. Although self-attention over a concatenated sequence can mix information across both streams, it 

39 

does so symmetrically and without an explicit, directed mechanism that conditions the 

exchange on which branch is querying. 

**Figure 2.6:** _Attention mechanism from IFC-Net (Cai et al., 2025)_ 

As seen in Figure 2.6, cross-attention-based fusion has been utilized as a more effective alternative to overcome this problem. Unlike self-attention, which computes relationships within a single feature set, cross-attention enables one branch to directly query information from the other, allowing for dynamic and bidirectional exchange. Cai et al. (2025) demonstrated that cross-attention fusion between CNN and Transformer components allows more effective interaction between local texture features and global contextual representations by proposing the IFC-Net model, leading to improved classification performance. This highlights the importance of adaptive feature fusion strategies in enhancing a model’s ability to discriminate between classes. 

Cross-attention becomes even more important in fine-grained classification tasks, where class differences often depend on subtle visual details. Badar et al. (2025), with the MSCAS-Net architecture, demonstrated that combining self-attention and cross-attention across multiple scales helps the model better capture these fine-grained features, leading to improved classification performance. This insight is especially relevant when it comes 

40 

to distinguishing between Grade Group 2 (Gleason 3+4) and Grade Group 3 (Gleason 4+3), where the key difference lies in the proportion of pattern 4 tissue. Making this distinction accurately requires the model to capture both localized architectural patterns and broader tissue-level context, highlighting the value of multi-scale and cross-attention mechanisms. 

Wang et al. (2025), in the original PDHD-Net, fused CNN and Swin Transformer features using window-based self-attention applied after concatenation. Although this allows information mixing within a single sequence, it treats both feature streams symmetrically and does not explicitly model the directed, asymmetric query–key–value exchange that cross-attention can provide between branches. This limitation shows up in their reported sensitivity values for GGG 2 and GGG 3, which were lower than for other grade groups, which is a sign that the fusion mechanism is not adequate for the boundary between the two. 

Building on the cross-attention principles established by Cai et al. (2025) and Badar et al. (2025), the present study replaces the window-based self-attention fusion in the ConvSwin3D encoder with a cross-attention module that enables direct, bidirectional interaction between CNN and Swin Transformer features. This modification is intended to improve the models’ ability to capture the subtle morphological differences that distinguish GGG 2 from GGG 3, directly addressing the core limitation identified in Wang et al. (2025)’s study. 

41 

## **G. GradCam & Explainability in Medical Artificial Intelligence** 

In recent years, artificial intelligence (AI) has continued to have a significant impact on medicine and healthcare, with applications across a wide range of fields, including radiology, cardiology, and mental health (Topol, 2018; Briganti & Moine, 2020). However, they also entail considerable risks, downsides, and negative implications; as such, there is a need to validate the reliability of AI tools, especially in high-stakes clinical settings. Additionally, as noted by Amann et al. (2020) and Hildt (2025), clinicians who cannot interpret a model’s outputs are limited in their ability to critically evaluate or take responsibility for AI-assisted decisions. As such, to solve this problem, the concept of explainability in medical AI becomes increasingly important as it refers to how well a system enables users to understand the reasoning behind its prediction. Without explainability, deep learning models function as “black boxes”, which makes the model difficult to trust, makes bias harder to detect, and reduces meaningful involvement of patients. This concern is further emphasized by Rudin (2019), while Antoniadi et al. (2021) identified a lack of interpretability as one of the most consistent barriers to clinical adoption of AI systems. 

Among explainability methods, Grad-CAM, introduced by Selvaraju et al. (2016), is widely used in medical imaging because it generates heatmaps that highlight regions most influential to the model’s predictions. Grad-CAM leverages gradients from the final convolutional layer, which makes it particularly suitable for spatially sensitive tasks. This spatial localization is especially important in imaging diagnosis, as noted by Chaddad et al. (2023). In prostate MRI, aligning model explanations with clinical frameworks such as the Prostate Imaging-Reporting and Data System (PI-RADS) ensures that predictions 

42 

are grounded in clinically meaningful features. For instance, Grad-CAM visualizations have demonstrated the ability to highlight lesion regions consistent with pathology and radiology findings, thereby enhancing interpretability and clinical trust (Al-Khanaty et al., 2025; Wang et al., 2025). 

Despite their utility, explainability methods possess notable limitations. Ghassemi et al. (2021) observed that heatmap-based techniques such as Grad-CAM indicate correlations rather than causation, which may result in overconfidence in model outputs. Nevertheless, there is broad agreement that some degree of interpretability is preferable to none, particularly in high-risk clinical applications (Amann et al., 2022; Rasheed et al., 

2022). Recent studies have evaluated Grad-CAM by having clinicians rate heatmaps using Likert scales. For example, Filvantorkaman et al. (2025) engaged five board-certified radiologists to assess Grad-CAM++ heatmaps on a 5-point scale, reporting mean scores of 4.4 for explanation usefulness and 4.0 for heatmap-region correspondence. 

## **H. Synthesis** 

The ability of advanced deep learning architectures to integrate complex feature extraction from MRI data enables the development of hybrid models, as demonstrated by the presented studies above, that overcome the limitations of traditional prostate cancer grading. This proves its suitability in accurately automating tumor classification and improving non-invasive diagnostics, highlighting its potential as a highly valuable tool in clinical oncology. 

43 

**Table 2.1:** _Synthesis Table_ 

|**Author(s) & Year**|**Title**|**Abstract**|**Findings**|
|---|---|---|---|
|Duran, A., Polson,<br>R., Martín-Isla, C.,<br>Dill, R., Vilanova,<br>J. C.,<br>Álvarez-Jiménez,<br>R., & Lekadir, K.<br>(2022)|ProstAttention-Net:<br>A Deep Attention<br>Model for Prostate<br>Cancer<br>Segmentation by<br>Aggressiveness in<br>MRI Scans|Proposed π-Net, an<br>attention-based<br>U-Net with dual<br>branches, designed<br>to simultaneously<br>segment and stratify<br>prostate cancer<br>lesions by<br>aggressiveness<br>level using MRI.|Achieved<br>69.0%±14.5%<br>sensitivity at 2.9<br>FP/patient (whole<br>prostate) and<br>70.8%±14.4% at<br>1.5 FP/patient<br>(peripheral zone<br>only),<br>outperforming<br>U-Net, Attention<br>U-Net,<br>DeepLabv3+, and<br>E-Net.|
|Jiang, Z., Ding, C.,<br>Liu, M., & Tao, D.<br>(2022)|SwinBTS: A<br>Method for 3D<br>Multimodal Brain<br>Tumor<br>Segmentation<br>Using Swin<br>Transformers|Proposed SwinBTS,<br>a 3D Swin<br>Transformer<br>encoder paired with<br>a CNN decoder for<br>volumetric brain<br>tumor segmentation<br>using multimodal<br>MRI.|Demonstrated that<br>Swin Transformers<br>significantly reduce<br>parameter count<br>and computational<br>complexity for 3D<br>volumetric tasks<br>compared to<br>standard ViTs,<br>without sacrificing<br>feature learning<br>capability.|
|He, Y., Nath, V.,<br>Yang, D., Tang, Y.,<br>Myronenko, A., &<br>Xu, D. (2023)|SwinUNETR-V2:<br>Stronger Swin<br>Transformers with<br>Staged-Convolution<br>al Feature<br>Projection for<br>Medical Image<br>Segmentation|Proposed<br>SwinUNETR-V2,<br>an enhanced Swin<br>Transformer<br>segmentation<br>network that<br>integrates staged<br>convolutional<br>feature projection<br>layers into the<br>encoder to improve<br>local feature<br>representation and<br>support|Demonstrated that<br>hierarchical Swin<br>Transformer<br>variants can project<br>heterogeneous MRI<br>data streams —<br>including T2W,<br>ADC, and DWI —<br>into a unified<br>feature space for<br>3D segmentation.|



44 

|||heterogeneous<br>multimodal MRI<br>inputs.||
|---|---|---|---|
|Fu, X., Bi, L.,<br>Kumar, A., Fulham,<br>M., & Kim, J.<br>(2024)|SSTrans-Net:<br>Learning Accurate<br>Target Distributions<br>With Swin<br>Transformers for<br>Medical Image<br>Segmentation|Proposed<br>SSTrans-Net, a<br>Swin<br>Transformer-based<br>segmentation<br>network designed to<br>learn accurate<br>spatial target<br>distributions for<br>lesions that span<br>large or irregular<br>regions in medical<br>images.|Established that<br>Swin Transformers<br>are particularly<br>effective at<br>modeling accurate<br>target distributions<br>for lesions that span<br>extensive areas — a<br>characteristic<br>relevant to prostate<br>tumors, which vary<br>widely in shape and<br>spatial extent.|
|Zheng, Y., Zhang,<br>J., Huang, D., Hao,<br>X., Qin, W., & Liu,<br>Y. (2024)|Detecting<br>MRI-Invisible<br>Prostate Cancers<br>Using a Weakly<br>Supervised Deep<br>Learning Model|Proposed<br>WS-UNet, a weakly<br>supervised 3D<br>U-Net that<br>leverages<br>voxel-level spatial<br>relationships and<br>patient-level labels<br>to detect prostate<br>cancers that are<br>imperceptible under<br>standard MRI<br>review.|Achieved an AUC<br>of 0.764 on the test<br>set. Demonstrated<br>that 3D spatial<br>volumetric<br>modeling captures<br>inter-slice<br>relationships<br>critical to detecting<br>lesions that are<br>invisible under<br>conventional 2D or<br>spatial-only<br>analysis.|
|Xu, Y., Lin, X., &<br>Zheng, Y. (2024)|Poisson Ordinal<br>Network for<br>Gleason Group<br>Estimation Using<br>Bi-Parametric MRI|Proposed a Poisson<br>Ordinal Network<br>(PON) combined<br>with a contrastive<br>learning memory<br>bank to estimate<br>Gleason Grade<br>Groups from<br>biparametric MRI.|Achieved superior<br>ordinal GGG<br>estimation across<br>GGG 1–5 compared<br>to standard<br>classification<br>baselines on<br>bpMRI.|
|Ewaidat, M. A.,<br>Duwairi, R., &|Frequency-Guided<br>U-Net: Leveraging<br>Attention Filter|Proposed a<br>Frequency-Guided<br>U-Net (GFNet) that|Achieved a Mean<br>Dice of 0.8366 and<br>Mean IoU of|



45 

|Al-Ayyoub, M.<br>(2024)|Gates and Fast<br>Fourier<br>Transformation for<br>Enhanced Medical<br>Image<br>Segmentation|integrates fast<br>Fourier transform<br>operations with<br>attention filter gates<br>into a standard<br>U-Net encoder.|0.7962 on left<br>atrium MRI<br>segmentation.|
|---|---|---|---|
|Luo, X., Liao, W.,<br>He, Y., Tang, F.,<br>Wu, M., Shen, Y.,<br>& Zhang, S. (2024)|A Frequency<br>Selection Network<br>for Medical Image<br>Segmentation|Proposed FSSN, a<br>medical image<br>segmentation<br>network<br>incorporating a<br>trainable frequency<br>band filter that<br>dynamically selects<br>and emphasizes<br>task-relevant<br>frequency<br>components.|Achieved more<br>accurate<br>segmentation than<br>representative<br>fixed-frequency and<br>spatial-only<br>baselines across<br>two public medical<br>imaging<br>benchmarks.|
|Ding, Z., Zhang, Z.,<br>Zhou, Y., & Liu, J.<br>(2024)|FI-NET: Feature<br>Interaction Network<br>for Fine-Grained<br>Medical Image<br>Classification|Proposed FI-NET, a<br>hybrid feature<br>interaction network<br>designed to<br>overcome the<br>limitations of<br>concatenation-base<br>d fusion in<br>CNN–Transformer<br>architectures.|Demonstrated that<br>merely<br>concatenating<br>convolutional and<br>attention features,<br>then applying<br>self-attention to the<br>combined sequence,<br>underutilizes the<br>complementary<br>roles of local and<br>global information.|
|Wang, X., Zhang,<br>N., Liu, H., Wang,<br>J., & Wang, W.<br>(2025)|Deep Learning<br>Algorithm for<br>Predicting Gleason<br>Grade Group Based<br>on Prostate MRI<br>Image Set|Proposed<br>PDHD-Net, a<br>dual-branch hybrid<br>architecture<br>integrating a<br>multi-stage 3D<br>CNN with a Swin<br>Transformer for<br>Gleason Grade<br>Group prediction<br>from mpMRI.|Achieved GGG=5<br>sensitivity of<br>96.7%, csPCa<br>sensitivity of 88.4%<br>at 1 FP/patient, and<br>AUC of 0.82<br>(GGG≥2 vs.<br>GGG=1).|



46 

|Cai, W., Chen, S.,<br>& Zhang, D. (2025)|FC-Net: Interactive<br>CNN and<br>Transformer for<br>Cross-Attention<br>Fusion in Medical<br>Image<br>Classification|Proposed IFC-Net,<br>a hybrid<br>architecture that<br>replaces<br>conventional<br>concatenation-base<br>d fusion with a<br>cross-attention<br>mechanism<br>enabling directed,<br>asymmetric<br>information<br>exchange between<br>CNN and<br>Transformer<br>branches.|Demonstrated that<br>cross-attention<br>fusion enables more<br>effective interaction<br>between local<br>texture features and<br>global contextual<br>representations than<br>concatenation-base<br>d or self-attention<br>fusion strategies,<br>leading to improved<br>classification<br>performance across<br>medical image<br>classification<br>benchmarks.|
|---|---|---|---|
|Badar, M., Farrukh,<br>M., Gao, Y., &<br>Qureshi, M. I.<br>(2025)|MSCAS-Net:<br>Multi-Scale Cross<br>and Self-Attention<br>Network for<br>Fine-Grained<br>Medical Image<br>Classification|Proposed<br>MSCAS-Net, a<br>Swin<br>Transformer-based<br>architecture that<br>combines<br>self-attention and<br>cross-attention<br>mechanisms across<br>multiple feature<br>scales for<br>fine-grained<br>medical image<br>classification.|Achieved best<br>performance on the<br>APTOS, DDR, and<br>IDRID diabetic<br>retinopathy grading<br>benchmarks.|



47 

## **Chapter III Research Design and Methodology** 

This chapter presents the methodology of the study, including the hypotheses, research method, research design, research instruments, sampling and data gathering procedures, and statistical treatment of data. It explains how the proposed GCALF-Net is developed, trained, and evaluated using the PI-CAI dataset, as well as how the performance of the baseline and modified models is statistically validated. 

## **A. Hypothesis** 

The study focused on developing GCALF-Net, a modified hybrid CNN–Swin Transformer architecture integrating a Learnable Frequency Filter and a Cross-Attention Fusion module into the baseline PDHD-Net framework, for non-invasive Gleason Grade Group classification using biparametric MRI from the PI-CAI dataset. Therefore, the study hypothesized that: 

(1) _H_ ₀: Replacing the fixed spherical frequency mask in the FDSF module with a Learnable Frequency Filter did not result in a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

_Hₐ_ : Replacing the fixed spherical frequency mask in the FDSF module with a Learnable Frequency Filter resulted in a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

(2) _H_ ₀: Replacing the window-based self-attention mechanism in the ConvSwin3D encoder with a Cross-Attention Fusion module did not result in a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

_Hₐ_ : Replacing the window-based self-attention mechanism in the ConvSwin3D encoder with a Cross-Attention Fusion module resulted in a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

(3) _H_ ₀: The integration of both the Learnable Frequency Filter and the Cross-Attention Fusion module in the proposed GCALF-Net did not produce a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

_Hₐ_ : The integration of both the Learnable Frequency Filter and the Cross-Attention Fusion module in the proposed GCALF-Net produced a statistically significant improvement in classification performance across all Gleason Grade Groups compared to the baseline PDHD-Net. 

(4) H₀: The Grad-CAM-generated heatmaps produced by the proposed GCALF-Net are not clinically acceptable based on PI-RADS version 2 criteria, as evaluated by three (3) practicing urologists using a Likert scale. 

49 

Hₐ: The Grad-CAM-generated heatmaps produced by the proposed GCALF-Net are clinically acceptable based on PI-RADS version 2 criteria, as evaluated by three (3) practicing urologists using a Likert scale. 

## **B. Research Methods** 

This study employed an experimental research design to test the hypotheses that the integration of a Learnable Frequency Filter (LFF) and a Cross-Attention Fusion (CAF) module into the baseline PDHD-Net architecture significantly improves Gleason Grade Group (GGG) classification performance, and that the resulting Grad-CAM 

visualizations are clinically acceptable based on PI-RADS version 2 criteria. The study utilized a different dataset from Wang et al. (2025), specifically the PI-CAI public dataset, which consists of three MRI sequences: the T2W, ADC, and DWI, instead of the four MRI sequences used in the original PDHD-Net architecture, namely T2W, ADC, DWI1000 and DWI3000. 

The experimental phase centers on testing the proposed architectural modifications against a known baseline, which is the original PDHD-Net of Wang et al. (2025), adapted to accept three-channel bpMRI input consisting of T2W, ADC, and DWI from the PI-CAI public dataset. Four model configurations were implemented for comparison: (1) the adapted baseline PDHD-Net, (2) PDHD-Net with the Learnable Frequency Filter (LFF) only, (3) PDHD-Net with the Cross-Attention Fusion (CAF) module only, and (4) the full proposed model, GCALF-Net, which integrates both LFF and CAF. All model variants were trained on the same PI-CAI dataset using identical 

preprocessing steps and evaluation protocols to ensure a fair comparison. To address the inherent class imbalance within the dataset, minority GGG classes, specifically GGG 4 

50 

and 5 were subjected to a class-weighted loss function was applied during training to ensure that high-risk lesions, specifically GGG 4 and 5 were not overshadowed by majority benign cases. 

The same training hyperparameters, including learning rate, batch size, number of epochs, and loss function, were applied across all configurations, with the exception of hyperparameters specific to the introduced modules, which have no counterparts in the baseline model. 

To ensure the model generalizes to unseen data, the study employs Stratified Patient-Level 5-Fold Cross-Validation. Patients are kept together within the same fold to prevent data leakage, and stratification is utilized to maintain the proportional distribution of each Gleason Grade Group across all folds. The model is trained on four folds and evaluated on the remaining fold, repeating five times so that each fold serves as the test set exactly once. This approach aligns with the validation strategy used by Wang et al. (2025) and ensures that performance estimates are robust and reproducible across multiple independent data splits. 

To evaluate the performance of each of the four model configurations,  several statistical standard medical imaging metrics were applied. This comparative analysis aimed to determine whether the proposed architectural modifications, whether individually and synergistically, yielded a statistically significant improvement in classification performance compared to the adapted baseline. 

The Grad-CAM heatmaps are then generated after the prediction results in GCALF-Net to visualize the specific anatomical regions within the MRI volume that most heavily influenced the model’s final classification decision. The application of 

51 

Grad-CAM provides a transparent interpretability that allows a practicing urologist to ensure that the model’s Gleason grade predictions are based on clinically relevant lesions rather than irrelevant tissue or imaging artifacts. 

## **C. Research Design** 

This study employs an ablation experimental design to systematically evaluate the individual and combined effects of two proposed architectural modifications on Gleason Grade Group classification performance. This methodology aligns with Wang et al. (2025), who validated each component of the original PDHD-Net by using nn-Detection as a baseline and sequentially incorporating ConvSwin3D, the Frequency-Band-Shunting Feature Enhancement (FMB) module, and the 3D Learnable Bi-Directional Feature Fusion Pyramid Network (3D-LDFPN) to isolate the contribution of each module to false-positive reduction and segmentation consistency. The present study adopts this progressive approach for the Learnable Frequency Filter and the Cross-Attention Fusion module, utilizing the complete PDHD-Net architecture adapted to biparametric MRI as the baseline rather than nn-Detection. 

## **1. Planning Phase** 

During the planning stage, the researchers identified the study objectives and formulated hypotheses to guide the research. To evaluate these hypotheses, the PI-CAI Grand Challenge public dataset was selected as the exclusive data source. This dataset consists of a large-scale, multi-center collection of biparametric MRI examinations, including T2-weighted imaging (T2W), Apparent Diffusion Coefficient (ADC) maps, and Diffusion-Weighted Imaging (DWI) sequences, each paired with biopsy-confirmed ISUP grade labels spanning 

52 

GGG 1 to 5. The PI-CAI dataset was selected for its standardized, publicly accessible ground-truth annotations and its comprehensive coverage of GGGs, which is essential for evaluating classification performance at the intermediate-risk boundary. The multi-center composition further enhances the external validity of the experimental findings. An ablation design with four distinct model variants (Models A through D) was implemented to isolate the contribution of each proposed modification prior to assessing their combined effect. 

## 2. **Pre-Processing of Dataset** 

The study will do a treatment of the dataset before entering the model as it utilizes the PI-CAI dataset over the Prostate-X2 from the study of Wang. Harmonization will be via N4 Bias Field Correction which enhances homogeneity from the different vendors that introduced lighting biases (Dovrou et al. 2023). This will be then followed by an isotropic sampling to 1 x 1 x 1mm spacing as different hospitals take image slicing differently. 

The original PI-CAI had a high resolution of 640 x 640 per slice. Processing a 3D stack that large is too computational intensive, therefore it will be spatially resampled to 256 x 256 by center-cropping, leaving only the prostate to be featured in the image. The 3D structure of the images is anisotropic and inconsistent as well, with varying depth (slice) depending on the patient scan. Standardization will be applied to the dataset to set the fixed depth to 32 slices 

53 

using linear interpolation. Finally, Z-score normalization will be applied to the pre-processed dataset to standardize pixel density. 

## 3. **Research Analysis** 

During the analysis phase, the structure and composition of the PI-CAI dataset were examined to inform both the preprocessing strategy and the experimental evaluation protocol. The dataset consists of 1,500 cases and 776 ISUP-annotated lesions. The class distribution for the annotated set is notably imbalanced: GGG 1 represents 40.1% of lesions, GGG 2 accounts for 33.5%, GGG 3 for 14.0%, GGG 4 for 5.3%, and GGG 5 for 7.1%. This significant imbalance, with lower-grade cases substantially outnumbering higher-grade cases, was identified as a critical factor necessitating methodological mitigation to prevent the model from favoring majority classes during training and to ensure that evaluation metrics accurately reflect performance across all grade groups. 

To address this imbalance, stratified random sampling is implemented at the patient level to ensure proportional representation of each Gleason Grade Group across all cross-validation folds. Patient-level sampling, rather than lesion or slice-level sampling, is used to prevent data leakage, which may occur if multiple slices from the same patient appear in both training and evaluation partitions, thereby artificially inflating performance estimates. The dataset is divided into five folds for cross-validation, with every patient case serving as a test case exactly once across the five iterations. 

54 

Within the training folds, a class-weighted loss function is employed to prevent the model from favoring majority classes.In class-weighted loss function, to ensure that high-risk lesions, specifically GGG 4 and 5 are not overshadowed by the majority benign cases, a weighted loss function is applied during the optimization phase. By assigning higher penalty weights to misclassifications within these critical groups, maintaining classification accuracy on high-risk lesions. 

## 3. **Designing the Model and Optimization Strategy** 

This study centers on the development of GCALF-Net, a modified version of the PDHD-Net architecture described by Wang et al. (2025), which incorporates two targeted architectural modifications. Four model variants are constructed and evaluated under identical experimental conditions to isolate the contribution of each modification. Model A serves as the baseline, consisting of the original PDHD-Net adapted to accept three-channel biparametric MRI input (T2W, ADC, DWI) instead of the original four-channel multiparametric MRI input (T2W, ADC, DWI1000, DWI3000). Model B incorporates the Learnable Frequency Filter (LFF), which replaces the fixed spherical frequency mask in the FDSF module with a trainable Fourier-domain weight matrix. This modification enables the model to optimize the frequency decomposition boundary during training, rather than relying on a predefined hyperparameter. Model C incorporates the Cross-Attention Fusion (CAF) module, which replaces the window-based self-attention fusion mechanism in the ConvSwin3D encoder with a cross-attention mechanism. This approach allows each branch, the CNN and the 

55 

Swin Transformer, to selectively attend to and retrieve relevant information from the other branch. Model D integrates both modifications, combining the Learnable Frequency Filter (LFF) and the Cross-Attention Fusion (CAF) module within a single unified architecture. 

The progressive ablation approach offers several advantages over alternative experimental designs. Compared to evaluating only the final combined model against the baseline, the ablation design provides evidence for the independent contribution of each modification, which is essential for understanding the mechanisms underlying architectural performance and for guiding future research. Compared to a full factorial design requiring all possible module combinations, this approach maintains a manageable number of experimental conditions while still isolating the effect of each proposed change (Fostiropoulos & Itti, 2023). The ablation design also maintains methodological continuity with Wang et al. (2025), who validated PDHD-Net through the same progressive module-addition strategy. All model variants share identical training hyperparameters — including learning rate, batch size, number of epochs, optimizer, and loss function — differing only in the targeted architectural component, so that any observed performance difference can be attributed to the specific modification under examination. The sole exception concerns hyperparameters intrinsic to newly introduced modules, such as the learnable weight initialization of the Frequency Filter and the number of attention heads in the Cross-Attention Fusion module, which have no counterparts in the baseline and therefore cannot be treated as equivalent. 

56 

## 4. **Implementation** 

All four model variants are implemented in Python using the PyTorch deep learning framework and are trained on a cloud-based GPU instance provided by Vast.AI, which supplies the computational resources required for volumetric deep learning on three-dimensional MRI data. Standard data handling and preprocessing are performed using NumPy and Pandas. Evaluation outputs, including Grad-CAM heatmaps and per-class confusion matrices, are visualized using Matplotlib. 

Model performance is evaluated using a set of metrics aligned with those reported by Wang et al. (2025), along with additional measures to address the class imbalance present in the dataset. A detailed discussion of these metrics, together with the inferential statistical procedures used to validate the results of the ablation study, is provided in the Statistical Treatment section. 

**Figure 3.1.** _Research Design Diagram_ 

57 

## **D. Research Instruments** 

## **a. Hardware** 

The study utilized cloud-based GPU computing resources from Vast.ai to meet the computational demands of training and evaluating the proposed deep learning architecture. The minimum hardware specifications required to perform the study include a GPU with at least 12 GB of VRAM (e.g., NVIDIA RTX 3060), a multi-core processor with at least 6 cores, 16 GB of system RAM, and sufficient solid-state storage for handling large-scale 3D MRI datasets, running on a Linux-based operating system (e.g., Ubuntu 22.04 LTS) with CUDA support 

(e.g., CUDA 11.8). Cloud-based infrastructure was selected over local hardware, given the volumetric nature of 3D biparametric MRI data and the significant memory and processing requirements of the hybrid CNN–Swin Transformer model. All model training, validation, and testing procedures were executed on the allocated GPU instances. 

## **b. Software** 

The primary programming language used throughout the study was Python, owing to its extensive ecosystem of scientific computing and deep learning libraries. PyTorch served as the deep learning framework for constructing and training the proposed architecture, ConvSwin3D, 3D-LDFPN, and the FDSF module. NumPy and SciPy provided the mathematical and matrix operations necessary to execute 3D Fast Fourier Transform (3D FFT) operations within the FDSF module. Matplotlib was used to generate all graphical outputs, 

58 

including training evaluation curves and Grad-CAM heatmaps produced for clinical interpretability assessment. 

## **c. People** 

Expert consultation was sought from three practicing urologists, who collectively served as the clinical validators for the study's interpretability evaluation. Each urologist independently assessed Grad-CAM heatmaps generated for 30 randomly selected stratified test cases — approximately 12 GGG 1, 10 GGG 2, 4 GGG 3, and 4 GGG 4 and 5 cases — using a five-point Likert scale to measure alignment with PI-RADS version 2 criteria. Specifically, the evaluation was based on the following three questions: 

1. To what extent do the high-intensity regions of the Grad-CAM heatmap correspond to clinically suspicious regions of interest (ROI)? 

2. How accurately does the heatmap delineate the boundaries of the suspected lesion without excessive extension into adjacent healthy prostate tissue? 

3. How clear and interpretable is the heatmap as a visual explanation for supporting the predicted Gleason Grade Group? 

All three evaluators were blinded to both the model's predictions and the 

corresponding ground-truth labels to ensure objectivity in the assessment. The use of three independent raters allows for inter-rater reliability analysis, thereby strengthening the validity of the clinical interpretability findings. Additional guidance was sought from the thesis adviser, whose expertise informed the 

59 

architectural design decisions, evaluation framework, and overall scientific rigor of the study. 

## **d. Data Source** 

The study drew exclusively on the PI-CAI Grand Challenge public dataset. The dataset comprises bpMRI scans (T2W, ADC, DWI), annotated with histopathologically confirmed GGG labels. This publicly available, multi-institutional dataset was selected to ensure reproducibility and to provide a sufficiently large and clinically representative sample for training and evaluating the proposed model across all four target grade groups. 

## **E. Sampling and Data Gathering Procedure** 

The PI-CAI Grand Challenge public dataset is the sole data source for this study. This resource comprises a large-scale, multi-center collection of bpMRI examinations, including T2W imaging, ADC maps, and DWI sequences, each paired with biopsy-confirmed ISUP grade labels. 

**Table 3.1:** _Characteristic of PI-CAI Grand Challenge Dataset_ 

|**Characteristic**|**Frequency**|
|---|---|
|Number of sites|11|
|Number of MRI scanners|5 S, 2 P|
|Number of patients|1476|
|Number of cases|1500|
|— Benign or indolent PCa|1075|
|— csPCa (ISUP ≥ 2)|425|



60 

|Median age (years)|66 (IQR:<br>61–70)|
|---|---|
|Median PSA (ng/mL)|8.5 (IQR: 6–13)|
|Median prostate volume (mL)|57 (IQR:<br>40–80)|
|Number of positive MRI lesions|1087|
|— PI-RADS 3|246 (23%)|
|— PI-RADS 4|438 (40%)|
|— PI-RADS 5|403 (37%)|
|Number of ISUP-based lesions|776|
|— ISUP 1|311 (40%)|
|— ISUP 2|260 (34%)|
|— ISUP 3|109 (14%)|
|— ISUP 4|41 (5%)|
|— ISUP 5|55 (7%)|



The class distribution of the PI-CAI dataset is presented in Table 3.0. GGG 1 accounts for 40.1% of lesions, GGG 2 for 33.5%, GGG 3 for 14.0%, GGG 4 for 5.3%, and GGG 5 for 7.1%, totaling 776 annotated lesions across 1,476 patients. This pronounced imbalance, with lower-grade cases substantially outnumbering higher-grade cases, necessitates the use of stratified sampling to preserve class proportions across all folds and the application of weighted loss functions to prevent the model from favoring majority classes during training. 

Stratified random sampling is implemented to address the inherent class imbalance in Gleason Grade Group distributions within prostate cancer datasets. Purely 

61 

random partitioning in imbalanced datasets can yield evaluation metrics that do not reliably reflect model performance, particularly for minority classes, as class imbalance has been shown to significantly affect the informativeness of global evaluation metrics in deep learning models for medical image classification (Hellín et al., 2024). This methodology aligns with prior studies in prostate cancer grading: Yang et al. (2026) utilized stratified random sampling to preserve proportional representation of grade groups across training, validation, and testing sets, while Pachetti and Colantonio (2023) stratified data splits by both lesion class and lesion location to ensure balanced representation across partitions. 

Sampling is performed at the patient level rather than at the lesion or slice level to prevent data leakage, which can occur when multiple slices or lesions from the same patient appear in both training and evaluation sets, thereby artificially inflating performance metrics. Pachetti and Colantonio (2023) also adopted a patient-wise splitting approach for this reason, emphasizing the necessity of strict separation between patients to ensure that evaluation results reflect true generalization performance. Assigning all data from a given patient to a single partition in this study similarly ensures that evaluation results accurately reflect the model's ability to generalize to previously unseen patients. 

The dataset is partitioned into five folds for cross-validation, consistent with the validation strategy used by Wang et al. (2025) in the original PDHD-Net study. In each iteration, four folds are used for training and one fold for evaluation, ensuring that each patient case serves as a test case exactly once. This approach aligns with the evaluation protocol of the baseline architecture, as Wang et al. (2025) also reported all FROC and 

62 

ROC results from five-fold cross-validation. The use of five-fold cross-validation is further supported by Pachetti and Colantonio (2023), who employed the same strategy to enhance the reliability of model evaluation and to obtain robust performance estimates with associated confidence intervals across multiple independent folds. Five-fold cross-validation is particularly suitable for this study because it maximizes data utilization for both training and evaluation, while generating five independent performance estimates that can be aggregated into robust summary statistics and subjected to formal statistical comparison. 

## **F. Statistical Treatment of the Data** 

The researchers used a variety of metrics to assess the performance of the four model configurations, to determine if there is a significant improvement in GCALF-Net compared to the baseline PDHD-Net, and to validate the clinical interpretability of the generated Grad-CAM heatmaps. 

To evaluate classification and localization performance, the following standard medical imaging metrics were utilized: Area Under the Receiver Operating Characteristic Curve (AUROC), Dice Similarity Coefficient (DSC), Sensitivity, Specificity, and Free-Response Receiver Operating Characteristic (FROC) analysis, Weighted F1-score, confusion matrix. To ensure the models generalize effectively to unseen data and maintain robustness across different data distributions, a Stratified Patient-Level 5-Fold Cross-Validation strategy was implemented. 

Furthermore, specific statistical measures were integrated into the training phase to mitigate the inherent class imbalance of the dataset. This included the use of a 

63 

Class-Weighted Cross-Entropy Loss function to ensure that the model prioritizes rare high-grade lesions equally with the majority classes. 

In addition to descriptive metrics, inferential statistical analyses were conducted to determine the significance of performance variances across the four configurations. 

Specifically, a One-Way Analysis of Variance (ANOVA) was employed, followed by Tukey’s Post-hoc test for pairwise comparisons. Finally, for the qualitative assessment of clinical interpretability, Grad-CAM visualizations were evaluated by three urologists using a five-point Likert scale to measure alignment with PI-RADS version 2 clinical criteria. 

## **a. Area Under the Receiver Operating Characteristic Curve (AUROC)** 

This metric measures how well the model distinguishes between the five Gleason Grade Group classes. Since AUROC is originally designed for binary classification, a One-vs-Rest (OvR) approach is applied for this multi-class task, where each GGG class is evaluated against all other classes. This allows for a clearer assessment of how well the model separates each grade group from the rest. The formula for AUROC is presented in (3.1) 

**==> picture [162 x 33] intentionally omitted <==**

Where: 

- TPR: True Positive Rate 

- FPR: False Positive Rate 

- u: The threshold value 

## **b. Dice Similarity Coefficient (DSC)** 

64 

The DSC measures the spatial overlap between the model’s predicted lesion segmentation maps and the ground-truth annotations provided by radiologists. This metric evaluates how accurately the model localizes suspicious regions within the prostate. The formula for DSC is presented in (3.2) 

**==> picture [71 x 20] intentionally omitted <==**

Where: 

- _X_ : Predicted segmentation volume 

- _Y_ _**:**_ Ground-truth annotation volume 

- ∩: Cardinality (size) of the volume 

## **c. Sensitivity** 

Sensitivity measures the True Positive Rate of lesion detection. In 

a multi-class context, this evaluates the model’s ability to correctly 

identify the presence of a specific Gleason grade. The formula for 

Sensitivity for each class _i_ (GGG 1–5), is presented in (3.3) 

**==> picture [113 x 26] intentionally omitted <==**

Where: 

- _TPi (True Positives):_ Instances of Actual GGG _i_ correctly classified 

as Predicted GGG _i_ 

- _FNi (False Negatives):_ Instances of Actual GGG _i_ incorrectly 

classified as any other GGG grade 

## **d. Specificity** 

65 

Specificity measures the model’s ability to correctly identify true 

negative cases. This metric is reported per class to determine how 

precisely the model identifies each Gleason grade and manages critical 

clinical boundaries. The formula for Specificity for each class i (GGG 

1–5), is presented in (3.4). 

**==> picture [115 x 26] intentionally omitted <==**

Where: 

- _TNi (True Negative):_ Instances of any grade other than GGG _i_ that 

were correctly not classified as GGG _i_ 

- _FPi (False Positives):_ Instances of any grade other than GGG _i_ that were incorrectly classified as GGG _i_ 

## **e. Free-Response Receiver Operating Characteristic (FROC) Analysis** 

FROC analysis is the clinical standard for evaluating lesion 

detection performance in prostate MRI. Unlike traditional image-level 

metrics, FROC is lesion-based and accounts for the model's ability to 

localize multiple malignant regions within a single image volume. The 

final FROC score is calculated as the mean sensitivity across a predefined set of average false positive (FP) thresholds per case. The formula for FROC analysis is presented in (3.5) and (3.6). 

The formula for the Sensitivity ( _S_ ) at a specific false positive threshold ( _k_ is: 

**==> picture [91 x 27] intentionally omitted <==**

66 

The overall FROC Score (Savg) used for model ranking is: 

**==> picture [96 x 25] intentionally omitted <==**

Where: 

- _S(k):_ The detection sensitivity at k average false positives per case. 

- _NTP (k):_ The number of correctly detected lesions (True Positives) at threshold $k$. A lesion is considered "detected" if its spatial overlap with the ground truth exceeds a defined Dice threshold 

(typically 0.1). 

- _Ntotal_lesions:_ The total count of malignant lesions in the ground-truth annotations. 

- _K:_ The set of operating points (Average False Positives per case). Following the Wang et al. and PI-CAI standards, these points are typically {0.11, 0.25, 0.5, 1, 2, 4, 8}. 

## **f. Weighted F1 Score** 

This metric is added due to the class imbalance in the PI-CAI 

dataset. This metric ensures that the model balances prediction and recall across all four classes simultaneously, rather than favoring the majority class. The formula for the Weighted F1 Score is presented in (3.7). 

**==> picture [222 x 34] intentionally omitted <==**

Where: 

**==> picture [220 x 26] intentionally omitted <==**

67 

**●** 𝑆𝑢𝑝𝑝𝑜𝑟𝑡 **:** Number of actual occurrences of class _i._ 𝑖 

## **g. Confusion Matrix** 

The Confusion Matrix provides a granular $5 \times 5$ evaluation of the four model configurations by mapping predicted GGG classifications against biopsy-confirmed ground truths. Utilizing a One-vs-All framework, this matrix enables a detailed analysis of error patterns that global metrics may obscure. The diagonal elements ( _TPi_ signify correct classifications, while off-diagonal cells identify specific diagnostic errors: those above the diagonal represent under-estimation (Type II error), and those below represent over-estimation (Type I error). This mapping is essential for validating how the LFF and CAF modules improve discrimination at critical clinical boundaries. 

68 

## **Table 3.2.** _5x5 Confusion Matrix_ 

||GGG 1|GGG 2|GGG 3|GGG 4|GGG 5|
|---|---|---|---|---|---|
|GGG 1|TP|||||
|GGG 2||TP||||
|GGG 3|||TP|||
|GGG 4||||TP||
|GGG 5|||||TP|



## **h. Stratified Patient-Level 5-Fold Cross-Validation strategy** 

This strategy is employed to ensure that the model’s performance 

is consistent and not dependent on a specific random split of the data. The dataset is partitioned into five distinct folds ( _k_ = 5), where each fold 

maintains a proportional representation of the five Gleason Grade Groups (Stratified) and ensures that all image slices from a single patient are kept 

together (Patient-Level) to prevent data leakage. The final reported 

performance is the average of the metrics calculated across all five iterations. 

## **i. Class-Weighted Cross-Entropy Loss function** 

To address the significant class imbalance in the dataset, a 

Class-Weighted Cross-Entropy loss function is utilized during the training 

69 

phase. This function modifies the standard cross-entropy objective by assigning a higher penalty (weight) to misclassifications of minority classes (high-grade lesions). This ensures that the model optimization 

process does not become biased toward the majority benign or low-grade classes. The formula for is represented in (3.8). 

**==> picture [118 x 34] intentionally omitted <==**

Where: 

- ℒ : The total weighted cross-entropy loss for a single sample 𝑊𝐶𝐸 

- _C:_ The total number of GGG classes ( _C_ = 5) 

- _wj_ : The assigned weight for class _j_ , inversely proportional to the class frequency 

- _yj_ : The ground-truth label (1 if the sample belongs to class _j_ , 0 otherwise) 

- ŷ _j_[:  The model’s predicted probability for class ] _[j]_[. ] 

## **j. Shapiro-Wilk** 

Before conducting comparative analyses, the Shapiro-Wilk test is 

applied to the metric distributions of each model. This test evaluates the null hypothesis that the data follows a normal distribution. The result 

determines whether parametric or non-parametric tests are used. The formula for this equation is represented in (3.9). 

**==> picture [81 x 49] intentionally omitted <==**

70 

Where: 

- _W_ :  The Shapiro-Wilk test statistic 

- _x(i)_ : The _i[th]_ smallest value in the sample (ordered values) 

- _ai_ : Constant coefficients derived from the covariances of the order 

   - statistics 

- _x̄_ : The mean of the sample values 

## **i. One-Way Repeated Measures Analysis of Variance (ANOVA)** 

If the data is normally distributed, a One-Way Repeated 

Measures Analysis of Variance (ANOVA) is conducted. This test 

determines if there is a significant difference between the means of 

the four models. The formula for this equation is represented in 

(3.10). 

**==> picture [72 x 25] intentionally omitted <==**

Where: 

- F: The F-ratio 

- _M Sbetween:_ Mean square variance between the model configurations 

- _M Serror_ : Mean square variance within the groups (error 

term) 

## **ii. Tukey’s Honestly Significant Difference (HSD)** 

Tukey’s HSD is the standard parametric post-hoc test used 

following a significant ANOVA. It compares the means of every 

possible model pair to identify where the specific significant 

71 

differences lie, while strictly controlling the family-wise error rate. 

This ensures that the probability of making a Type I error (finding a difference where none exists) remains at 5% across all 

comparisons. The formula for this equation is represented in 

- (3.11). 

**==> picture [97 x 22] intentionally omitted <==**

Where: 

- _q_ : The studentized range statistic, found in a standard 

   - q-table based on the number of groups and degrees of freedom 

- _M Serro_ r: The Mean Square Error derived from the preceding ANOVA calculation 

- _n_ : The number of observations per group (in this study, _n_ = 5 folds) 

## **iii. Friedman Test** 

If the data violates the assumption of normality, the 

Friedman Test is used. This is a rank-based alternative to ANOVA 

used to detect differences across multiple test attempts. The 

formula for this equation is represented in (3.12). 

**==> picture [182 x 35] intentionally omitted <==**

Where: 

**==> picture [170 x 13] intentionally omitted <==**

72 

- _n_ : The number of folds (n = 5) 

- _k_ : The number of model configurations (k = 4) 

- _Rj_ : The sum of ranks for the j[th] model 

## **iv. Wilcoxon Signed-Rank Test (with Bonferroni Correction)** 

The Wilcoxon Signed-Rank test is the non-parametric 

alternative used when the Shapiro-Wilk test indicates a non-normal distribution. Since it is used for "related" samples (the four models running on the same 5 folds), it compares the median ranks of the performance metrics. Because multiple comparisons are being 

made, a Bonferroni Correction is applied to the significance level ( _a_ ) to maintain statistical integrity. The formula for this equation is represented in (3.13). 

**==> picture [144 x 31] intentionally omitted <==**

Where: 

- _sgn_ : The sign function (returns +1, -1, or 0 depending on whether the difference between model pairs is positive, negative, or zero). 

- _x1,i,x2,i:_ The performance scores of the two models being compared in fold _i._ 

- _Ri_ : The rank of the absolute difference |x1,i - x2,i| 

- _n_ : The number of folds ( _n_ = 5) 

## **k. Five-Point Likert Scale** 

73 

To evaluate clinical interpretability, Grad-CAM heatmaps are generated for the combined model's predictions on the test set. The heatmap is generated for the predicted class and overlaid on the original T2W MRI sequence. A sample of 30 test set cases, selected using 

stratified random sampling to preserve class distribution, is used for evaluation. 

**Table 3.3:** _Five-point Likert Scale_ 

|Score|Label|Description|
|---|---|---|
|1|Very Poor|Heatmap highlights regions completely unrelated to PI-RADS criteria|
|2|Poor|Minimal alignment; primarily non-relevant areas|
|3|Fair|Moderate alignment; misses key features or includes substantial<br>irrelevant areas|
|4|Good|Close match with expected lesion location and PI-RADS features, with<br>minor deviations|
|5|Excellent|Precise outline directly corresponding to PI-RADS-relevant anatomical<br>or diffusion features|



As seen in Table 3.3, a practicing urologist rates each heatmap 

using a five-point Likert scale measuring alignment with PI-RADS version 2 clinical criteria. To prevent bias in the ratings, the urologist does not know the model's predicted Gleason grade or the ground-truth labels when rating the heatmaps.The mean Likert score and standard deviation are calculated from all rated test cases. A mean score of 4.0 or higher is considered evidence that the model's explanations are clinically meaningful. This threshold follows Filvantorkaman et al. (2025), who 

74 

reported mean scores of 4.4 for explanation usefulness and 4.0 for heatmap-region correspondence using a similar five-point scale with board-certified radiologists. 

75 

## **References** 

Al-Khanaty, A., Hennes, D., Guduguntla, A., Guerrero, P., Delgado, C., Dinneen, E., 

Mazzone, E., Appu, S., Bolton, D., Eapen, R. S., Murphy, D. G., Lawrentschuk, N., & Perera, M. L. (2025). Using artificial intelligence as a risk prediction model in patients with equivocal multiparametric prostate MRI findings. _Cancers_ , _18_ (1), 28. https://doi.org/10.3390/cancers18010028 

Amann, J., Vetter, D., Blomberg, S. N., Christensen, H. C., Coffee, M., Gerke, S., Gilbert, T. K., Hagendorff, T., Holm, S., Livne, M., Spezzatti, A., Strümke, I., Zicari, R. 

V., Madai, V. I., & Initiative, O. B. O. T. Z. (2022). To explain or not to explain?—Artificial intelligence explainability in clinical decision support systems. _PLOS Digital Health_ , _1_ (2), e0000016. 

https://doi.org/10.1371/journal.pdig.0000016 

Badar, D., Abbas, J., Alsini, R., Abbas, T., ChengLiang, W., & Daud, A. (2025). Transformer attention fusion for fine grained medical image classification. _Scientific Reports_ , _15_ (1), 20655. https://doi.org/10.1038/s41598-025-07561-x 

Badar, M., Malik, H., & Imran, A. S. (2025). MSCAS-Net: Multi-scale cross-attention and self-attention network for fine-grained classification. _Applied Intelligence_ . https://doi.org/10.1007/s10489-025-05981-0 

Baliyan, V., Das, C. J., Sharma, R., & Gupta, A. K. (2016). Diffusion weighted imaging: Technique and applications. World Journal of Radiology, 8(9), 785. https://doi.org/10.4329/wjr.v8.i9.785 

76 

Bansal, C., Anand, G., & Goyal, A. (2025). Evolution of Gleason score prostate — a 

review. _Indian Journal of Surgical Oncology_ , _16_ (6), 1538–1545. https://doi.org/10.1007/s13193-025-02251-6 

Bao, J., Hou, Y., Qin, L., Zhi, R., Wang, X.-M., Shi, H.-B., Sun, H.-Z., Hu, C.-H., & Zhang, Y.-D. (2023). High-throughput precision MRI assessment with integrated stack-ensemble deep learning can enhance the preoperative prediction of prostate cancer Gleason grade. _British Journal of Cancer_ , _128_ , 1267–1277. 

https://doi.org/10.1038/s41416-022-02134-5 

Borghesi, M., Ahmed, H., Nam, R., Schaeffer, E., Schiavina, R., Taneja, S., Weidner, W., & Loeb, S. (2017). Complications after systematic, random, and image-guided prostate biopsy. _European Urology_ , _71_ , 353–365. 

Bray, F., Laversanne, M., Sung, H., Ferlay, J., Siegel, R. L., Soerjomataram, I., & Jemal, A. (2024). Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 74(3), 229–263. https://doi.org/10.3322/caac.21834 

Briganti, G., & Moine, O. L. (2020). Artificial intelligence in medicine: Today and 

tomorrow. _Frontiers in Medicine_ , _7_ , 27. https://doi.org/10.3389/fmed.2020.00027 

Brunese, L., Mercaldo, F., Reginelli, A., & Santone, A. (2020). Radiomics for Gleason score detection through deep learning. _Sensors_ , _20_ (18), 5411. 

https://doi.org/10.3390/s20185411 

77 

Cai, S., Zhang, Q., Wang, S., Hu, J., Zeng, L., & Li, K. (2025). Interactive CNN and 

‐ ‐ Transformer based cross attention fusion network for medical image classification. _International Journal of Imaging Systems and Technology_ , _35_ (3). https://doi.org/10.1002/ima.70077 

Cao, X., Yan, J., & Xu, G. (2023). PFCA-Net: A post-fusion based cross-attention model for predicting PCa Gleason Group using multiparametric MRI. _2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ , 3507–3513. https://doi.org/10.1109/BIBM58861.2023.10385606 

Chen, Y., Lin, Y., Xu, X., Ding, J., Li, C., Zeng, Y., Xie, W., & Huang, J. (2022). Multi-domain medical image translation generation for lung image classification based on generative adversarial networks. _Computer Methods and Programs in Biomedicine_ , _229_ , 107200. https://doi.org/10.1016/j.cmpb.2022.107200 

Chu, F., Chen, L., Guan, Q., Chen, Z., Ji, Q., Ma, Y., Ji, J., Sun, M., Huang, T., Song, H., Zhou, H., Lin, X., & Zheng, Y. (2025). Global burden of prostate cancer: age-period-cohort analysis from 1990 to 2021 and projections until 2040. World Journal of Surgical Oncology, 23(1), 98. 

https://doi.org/10.1186/s12957-025-03733-1 

Chu, P., & Beasley, J. (1998). A genetic algorithm for the multidimensional knapsack 

problem. _Journal of Heuristics_ , _4_ (1), 63–86. 

https://doi.org/10.1023/A:1009642405419 

78 

de Vente, C., Vos, P., Hosseinzadeh, M., Pluim, J., & Veta, M. (2021). Deep learning 

regression for prostate cancer detection and grading in bi-parametric MRI. _IEEE Transactions on Biomedical Engineering_ , _68_ (2), 374–383. 

https://doi.org/10.1109/TBME.2020.2993528 

‐ Ding, Y., Liu, J., He, Y., Huang, J., Liang, H., Yi, Z., & Wang, Y. (2024). FI NET: Rethinking feature interactions for medical image segmentation. _Advanced Intelligent Systems_ , _6_ (12). https://doi.org/10.1002/aisy.202400201 

Dovrou, A., Nikiforaki, K., Zaridis, D., Manikis, G. C., Mylona, E., Tachos, N., Tsiknakis, M., Fotiadis, D. I., & Marias, K. (2023). A segmentation-based method improving the performance of N4 bias field correction on T2weighted MR imaging data of the prostate. Magnetic resonance imaging, 101, 1–12. https://doi.org/10.1016/j.mri.2023.03.012 

Duran, A., Dussert, G., Rouvière, O., Jaouen, T., Jodoin, P.-M., & Lartizien, C. (2022). ProstAttention-Net: A deep attention model for prostate cancer segmentation by aggressiveness in MRI scans. _Medical Image Analysis_ , _77_ , 102347. https://doi.org/10.1016/j.media.2021.102347 

Eble, J. N., Sauter, G., Epstein, J. E., et al. (2004). Tumors of the prostate. In _Pathology and genetics of the urinary system and male genital organs_ (pp. 159–215). IARC Press. 

Epstein, J. I., Allsbrook, W. C., Jr., Amin, M. B., Egevad, L. L., et al. (2005). The 2005 International Society of Urological Pathology (ISUP) Consensus Conference on 

79 

Gleason grading of prostatic carcinoma. _American Journal of Surgical Pathology_ , _29_ (9), 1228–1242. https://doi.org/10.1097/01.pas.0000173646.99337.b1 

Ewaidat, H. A., Brag, Y. E., E'layan, A. W. Y., & Almakhadmeh, A. (2024). 

_Frequency-Guided U-Net: Leveraging attention filter gates and fast Fourier_ 

_transformation for enhanced medical image segmentation_ . arXiv. https://arxiv.org/abs/2405.00683 

Fawcett, T. (2005). An introduction to ROC analysis. _Pattern Recognition Letters_ , _27_ (8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010 

Filvantorkaman, M., Piri, M., Torkaman, M. F., Zabihi, A., & Moradi, H. (2025, August 9). Fusion-Based brain tumor classification using deep learning and explainable AI, and Rule-Based reasoning. arXiv.org. https://arxiv.org/abs/2508.06891 

Fiorentino, V., Martini, M., Dell'Aquila, M., Musarra, T., Orticelli, E., Larocca, L. M., 

Rossi, E., Totaro, A., Pinto, F., Lenci, N., Di Paola, V., Manfredi, R., Bassi, P. F., & Pierconti, F. (2020). Histopathological ratios to predict Gleason score agreement between biopsy and radical prostatectomy. _Diagnostics_ , _11_ (1), 10. https://doi.org/10.3390/diagnostics11010010 

Fostiropoulos, I., & Itti, L. (2023). _ABLATOR: Robust horizontal-scaling of machine learning ablation experiments_ . AutoML 2023 Apps, Benchmarks, Challenges, and Datasets Track. 

https://proceedings.mlr.press/v224/fostiropoulos23a/fostiropoulos23a.pdf 

80 

Fu, L., Chen, Y., Ji, W., & Yang, F. (2024). SSTrans-Net: Smart Swin Transformer Network for medical image segmentation. _Biomedical Signal Processing and Control_ , _91_ , 106071. https://doi.org/10.1016/j.bspc.2024.106071 

Gader, T. B. A., Lachouak, H., Touil, S., Echi, A. K., & Mbarek, S. (2024). Hybrid CNN–Swin Transformer for glaucoma screening in fundus images. In _2024 IEEE/ACS 21st International Conference on Computer Systems and Applications (AICCSA)_ (pp. 1–8). IEEE. 

https://doi.org/10.1109/AICCSA63423.2024.10912542 

Gleason, D. F. (1977). Histologic grading and clinical staging of prostate carcinoma. In 

- M. Tannenbaum (Ed.), _Urologic pathology: The prostate_ (pp. 171–198). Lea & Fibiger. 

Gordetsky, J., & Epstein, J. (2016). Grading of prostatic adenocarcinoma: Current state and prognostic implications. _Diagnostic Pathology_ , _11_ (1), 25. https://doi.org/10.1186/s13000-016-0478-2 

Hamm, C. A., Baumgärtner, G. L., Biessmann, F., Beetz, N. L., Hartenstein, A., Savic, L. 

- J., Froböse, K., Dräger, F., Schallenberg, S., Rudolph, M., Baur, A. D. J., Hamm, 

- B., Haas, M., Hofbauer, S., Cash, H., & Penzkofer, T. (2023). Interactive Explainable deep learning model informs prostate cancer diagnosis at MRI. Radiology, 307(4), e222276. https://doi.org/10.1148/radiol.222276 

Haruna, Y., Qin, S., Chukkol, A. H. A., Yusuf, A. A., Bello, I., & Lawan, A. (2025). Exploring the synergies of hybrid convolutional neural network and Vision 

81 

Transformer architectures for computer vision: A survey. _Engineering Applications of Artificial Intelligence_ , _144_ , 110057. 

https://doi.org/10.1016/j.engappai.2025.110057 

He, Y., Nath, V., Yang, D., Tang, Y., Myronenko, A., & Xu, D. (2023). SwinUNETR-V2: Stronger Swin Transformers with stagewise convolutions for 3D medical image segmentation. _Lecture Notes in Computer Science_ , 416–426. 

https://doi.org/10.1007/978-3-031-43901-8_40 

Hellín, C. J., Olmedo, A. A., Valledor, A., Gómez, J., López-Benítez, M., & Tayebi, A. (2024). Unraveling the impact of class imbalance on deep-learning models for medical image classification. _Applied Sciences_ , _14_ (8), 3419. 

https://doi.org/10.3390/app14083419 

Hugosson, J., Godtman, R. A., Wallstrom, J., Axcrona, U., Bergh, A., Egevad, L., Geterud, K., Khatami, A., Socratous, A., Spyratou, V., Svensson, L., Stranne, J., Månsson, M., & Hellstrom, M. (2024). Results after four years of screening for prostate cancer with PSA and MRI. _New England Journal of Medicine_ , _391_ (12), 1083–1095. https://doi.org/10.1056/nejmoa2406050 

International Agency for Research on Cancer. (2024). _Philippines: Fact sheet (Globocan 2022)_ . World Health Organization. 

https://gco.iarc.who.int/today/data/factsheets/populations/608-philippines-fact-she et.pdf 

82 

Jiang, Y., Zhang, Y., Lin, X., Dong, J., Cheng, T., & Liang, J. (2022). SwinBTS: A 

method for 3D multimodal brain tumor segmentation using Swin Transformer. _Brain Sciences_ , _12_ (6), 797. https://doi.org/10.3390/brainsci12060797 

Jin, L., Yu, Z., Gao, F., & Li, M. (2024). T2-weighted imaging-based deep-learning method for noninvasive prostate cancer detection and Gleason grade prediction: A multicenter study. _Insights Into Imaging_ , _15_ (1), 111. 

https://doi.org/10.1186/s13244-024-01682-z 

Khatab, Z., Hanna, K., Rofaeil, A., Wang, C., Maung, R., & Yousef, G. M. (2023). Pathologist workload, burnout, and wellness: Connecting the dots. _Critical Reviews in Clinical Laboratory Sciences_ , _61_ (4), 254–274. 

https://doi.org/10.1080/10408363.2023.2285284 

- Ko, L., Gravina, N., Berghausen, J., & Abdo, J. (2025). Rising trends in prostate cancer among Asian men: Global concerns and diagnostic solutions. _Cancers_ , _17_ (6), 1013. https://doi.org/10.3390/cancers17061013 

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , _86_ (11), 2278–2324. https://doi.org/10.1109/5.726791 

Leenders, G. J. L. H., Kwast, T. H., Grignon, D. J., Evans, A. J., Kristiansen, G., 

Kweldam, C. F., Litjens, G., McKenney, J. K., Melamed, J., Mottet, N., Paner, G. 

P., Samaratunga, H., Schoots, I. G., Simko, J. P., Tsuzuki, T., Varma, M., Warren, A. Y., Wheeler, T. M., Williamson, S. R., Iczkowski, K. A., … ISUP Grading 

83 

Workshop Panel Members. (2020). The 2019 International Society of Urological Pathology (ISUP) Consensus Conference on Grading of Prostatic Carcinoma. _The American Journal of Surgical Pathology_ , _44_ (8), e87–e99. 

https://doi.org/10.1097/PAS.0000000000001497 

Li, Y. (2025, June 5). _AUROC and AUPRC_ . Medium. 

https://medium.com/@yukims19/auroc-and-auprc-b25183827e5a 

Li, Y., Wang, Y., Leng, T., & Zhijie, W. (2020). Wavelet U-Net for medical image segmentation. In _Artificial Neural Networks and Machine Learning – ICANN 2020_ (pp. 800–810). Springer. https://doi.org/10.1007/978-3-030-61609-0_63 

Liang, J., Cao, J., Sun, G., Zhang, K., Gool, V., & Timofte, R. (2021). SwinIR: Image restoration using Swin Transformer. _arXiv_ . https://arxiv.org/abs/2108.10257 

Linkon, A. H. M., Labib, M. M., Hasan, T., Hossain, M., & Jannat, M. (2021). Deep learning in prostate cancer diagnosis and Gleason grading in histopathology images: An extensive study. _Informatics in Medicine Unlocked_ , _24_ , 100582. https://doi.org/10.1016/j.imu.2021.100582 

Liu, F., Zhao, Y., Song, J., Tu, G., Liu, Y., Peng, Y., Mao, J., Yan, C., & Wang, R. (2024). A hybrid classification model with radiomics and CNN for high and low grading of prostate cancer Gleason score on mp-MRI. _Displays_ , _83_ , 102703. https://doi.org/10.1016/j.displa.2024.102703 

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. In 

84 

_Proceedings of the IEEE/CVF International Conference on Computer Vision_ 

_(ICCV)_ . 

https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_ Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.p 

df 

Luo, X., Wang, Y., & Zhang, S. (2024). A frequency selection network for medical image 

segmentation. _Heliyon_ , _10_ (16), e35698. 

https://doi.org/10.1016/j.heliyon.2024.e35698 

Melia, J., Moseley, R., Ball, R. Y., et al. (2006). A UK-based investigation of inter- and 

intra-observer reproducibility of Gleason grading of prostatic biopsies. 

_Histopathology_ , _48_ (6), 644–654. 

https://doi.org/10.1111/j.1365-2559.2006.02393.x 

MIMS Oncology Honorary Editorial Advisory Board. (2025, June 23). _Prostate cancer_ 

_disease background_ . MIMS. 

https://www.mims.com/philippines/disease/prostate-cancer/disease-background 

Mirabella, A. G. (2025). _Frequency domain analysis_ . 

- https://agiulianomirabella.github.io/teaching tutorials/frequency_domain_analysis .html 

Nagpal, K., Foote, D., Liu, Y., Chen, P. C., Wulczyn, E., Tan, F., Olson, N., Smith, J. L., Mohtashamian, A., Wren, J. H., Corrado, G. S., MacDonald, R., Peng, L. H., Amin, M. B., Evans, A. J., Sangoi, A. R., Mermel, C. H., Hipp, J. D., & Stumpe, 

85 

M. C. (2019). Development and validation of a deep learning algorithm for improving Gleason scoring of prostate cancer. _npj Digital Medicine_ , _2_ (1), 48. https://doi.org/10.1038/s41746-019-0112-2 

Nicola, R., & Bittencourt, L. K. (2023). PI-RADS 3 lesions: A critical review and discussion of how to improve management. _Abdominal Radiology_ , _48_ (7), 2401–2405. https://doi.org/10.1007/s00261-023-03929-7 

Ozkan, T. A., Eruyar, A. T., Cebeci, O. O., Memik, O., Ozcan, L., & Kuskonmaz, I. (2016). Interobserver variability in Gleason histological grading of prostate cancer. _Scandinavian Journal of Urology_ , _50_ (6), 420–424. 

https://doi.org/10.1080/21681805.2016.1206619 

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., 

Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In _Advances in Neural Information Processing Systems_ (Vol. 32). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9 f7012727740-Paper.pdf 

Patel, P., Wang, S., & Siddiqui, M. M. (2019). The use of multiparametric magnetic 

resonance imaging (MPMRI) in the detection, evaluation, and surveillance of clinically significant prostate cancer (CSPCA). _Current Urology Reports_ , _20_ (10), 60. https://doi.org/10.1007/s11934-019-0926-0 

86 

Perera, M., Assel, M., Nalavenkata, S., Khaleel, S., Benfante, N., Carlsson, S. V., Reuter, 

V. E., Laudone, V. P., Scardino, P. T., Touijer, K. A., Eastham, J. A., Vickers, A. J., Fine, S. W., & Ehdaie, B. (2024). Quantification of Gleason Pattern 4 Metrics identifies pathologic progression in patients with grade Group 2 prostate cancer on active surveillance. Clinical Genitourinary Cancer, 22(6), 102204. 

https://doi.org/10.1016/j.clgc.2024.102204 

PhilHealth. (2024). _Annex B: List of procedure case rates (Circular No. 2024-0037)_ . 

Philippine Health Insurance Corporation. 

https://www.philhealth.gov.ph/circulars/2024/0037/AnnexB-ListofProcedureCase Rates.pdf 

Prabhu, C. S., Gavade, A. B., Gavade, P. A., & Nerli, R. B. (2026). A review on 

innovations in prostate cancer diagnosis: Automated techniques for Gleason score estimation via mpMRI and DWSI imaging. _International Journal on Smart_ 

_Sensing and Intelligent Systems_ , _19_ (1). https://doi.org/10.2478/ijssis-2026-0001 

Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). _Global filter networks for image classification_ . arXiv. https://arxiv.org/abs/2107.00645 

Rasheed, K., Qayyum, A., Ghaly, M., Al-Fuqaha, A., Razi, A., & Qadir, J. (2022). 

Explainable, trustworthy, and ethical machine learning for healthcare: A survey. 

_Computers in Biology and Medicine_ , _149_ , 106043. 

https://doi.org/10.1016/j.compbiomed.2022.106043 

87 

Richard, P. O., Timilshina, N., Komisarenko, M., Martin, L., Ahmad, A., Alibhai, S. M., Hamilton, R. J., Kulkarni, G., & Finelli, A. (2020). The long-term outcomes of Gleason grade groups 2 and 3 prostate cancer managed by active surveillance: Results from a large population-based cohort. Canadian Urological Association Journal, 14(6), 174–181. https://doi.org/10.5489/cuaj.6328 

Saha, A., Bosma, J. S., Twilt, J. J., van Ginneken, B., Bjartell, A., Padhani, A. R., Bonekamp, D., Villeirs, G., Salomon, G., Giannarini, G., Kalpathy-Cramer, J., Barentsz, J., Maier-Hein, K. H., Rusu, M., Rouvière, O., van den Bergh, R., Panebianco, V., Kasivisvanathan, V., Obuchowski, N. A., Yakar, D., … PI-CAI Consortium. (2024). Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): An international, paired, non-inferiority, confirmatory study. The Lancet Oncology, 25(7), 879–887. 

Schatten, H. (2018). Brief overview of prostate cancer statistics, grading, diagnosis and treatment strategies. _Advances in Experimental Medicine and Biology_ , _1095_ , 1–14. https://doi.org/10.1007/978-3-319-95693-0_1 

Schlenker, B., Apfelbeck, M., Armbruster, M., Chaloupka, M., Stief, C. G., & Clevert, D.-A. (2019). Comparison of PIRADS 3 lesions with histopathological findings after MRI-fusion targeted biopsy of the prostate in a real world-setting. _Clinical Hemorheology and Microcirculation_ , _71_ (2), 165–170. https://doi.org/10.3233/ch-189407 

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based 

88 

localization. In _Proceedings of the IEEE International Conference on Computer Vision 2017_ (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74 

Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality 

(complete samples). _Biometrika_ , _52_ (3–4), 591–611. 

https://doi.org/10.1093/biomet/52.3-4.591 

Shen, Y., & Wu, S. (2023). Value of dynamic contrast-enhanced MRI and DWI in diagnosis of prostate cancer and its pathological comparison. _Journal of Men's Health_ , _19_ (12), 119. https://doi.org/10.22514/jomh.2023.138 

Short, E., Warren, A. Y., & Varma, M. (2019). Gleason grading of prostate cancer: A pragmatic approach. _Diagnostic Histopathology_ , _25_ (10), 371–378. https://doi.org/10.1016/j.mpdhp.2019.07.001 

Singh, O., & Bolla, S. R. (2023, July 17). _Anatomy, abdomen and pelvis, prostate_ . StatPearls Publishing. https://www.statpearls.com/point-of-care/27831 

Singh, Y., Farrelly, C., Hathaway, Q. A., Choudhary, A., Carlsson, G., Erickson, B., & Leiner, T. (2023). The role of geometry in convolutional neural networks for medical imaging. _Mayo Clinic Proceedings: Digital Health_ , _1_ (4), 519–526. https://doi.org/10.1016/j.mcpdig.2023.08.006 

Stefan, D. C., & Tang, S. (2023). Addressing cancer care in low- to middle-income 

countries: a call for sustainable innovations and impactful research. BMC Cancer, 

23(1), 756. https://doi.org/10.1186/s12885-023-11272-9 

89 

Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image 

segmentation: Analysis, selection, and tool. _BMC Medical Imaging_ , _15_ (1), 29. https://doi.org/10.1186/s12880-015-0068-x 

Takahashi, S., Sakaguchi, Y., Kouno, N., Takasawa, K., Ishizu, K., Akagi, Y., Aoyama, 

R., Teraya, N., Bolatkan, A., Shinkai, N., Machino, H., Kobayashi, K., Asada, K., Komatsu, M., Kaneko, S., Sugiyama, M., & Hamamoto, R. (2024). Comparison of vision transformers and convolutional neural networks in medical image analysis: A systematic review. _Journal of Medical Systems_ , _48_ (1). 

https://doi.org/10.1007/s10916-024-02105-8 

Tamada, T., Kido, A., Ueda, Y., Takeuchi, M., Fukunaga, T., Sone, T., & Yamamoto, A. 

(2021). Clinical impact of ultra-high b-value (3000 s/mm2) diffusion-weighted magnetic resonance imaging in prostate cancer at 3T: comparison with b-value of 2000 s/mm2. British Journal of Radiology, 95(1131), 20210465. 

https://doi.org/10.1259/bjr.20210465 

Topol, E. J. (2018). High-performance medicine: The convergence of human and 

artificial intelligence. _Nature Medicine_ , _25_ (1), 44–56. 

https://doi.org/10.1038/s41591-018-0300-7 

Venderink, W., Govers, T. M., De Rooij, M., Fütterer, J. J., & Sedelaar, J. P. M. (2017). 

Cost-effectiveness comparison of imaging-guided prostate biopsy techniques: 

Systematic transrectal ultrasound, direct in-bore MRI, and image fusion. 

_American Journal of Roentgenology_ , _208_ (5), 1058–1063. 

https://doi.org/10.2214/ajr.16.17322 

90 

Wang, C., He, Y., Wang, Y., Deng, Z., Wang, L., & Li, D. (2026). Vessel-density distribution mapping on portal venous CT and deep learning for 2-year overall survival risk stratification after transarterial chemoembolization in hepatocellular carcinoma. _Abdominal Radiology_ . https://doi.org/10.1007/s00261-026-05505-1 

Wang, W., Chen, Y., Shen, Y., & Zhang, S. (2024). FreMIM: Fourier transform meets masked image modeling for medical image segmentation. In _Proceedings of the 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ (pp. 7845–7855). IEEE. https://doi.org/10.1109/WACV57701.2024.00768 

Wang, X., Zhang, N., Liu, H., Wang, J., & Wang, W. (2025). Deep learning algorithm for predicting Gleason grade group based on prostate MRI image set. _Research Square_ . https://doi.org/10.21203/rs.3.rs-6818189/v1 

Wang, Y., Xin, Y., Zhang, B., Pan, F., Li, X., Zhang, M., Yuan, Y., Zhang, L., Ma, P., Guan, B., & Zhang, Y. (2025). Assessment of prostate cancer aggressiveness through the combined analysis of prostate MRI and 2.5D deep learning models. _Frontiers in Oncology_ , _15_ , 1539537. https://doi.org/10.3389/fonc.2025.1539537 

Werner, R. A., Hartrampf, P. E., Fendler, W. P., Serfling, S. E., Derlin, T., Higuchi, T., 

Pienta, K. J., Gafita, A., Hope, T. A., Pomper, M. G., Eiber, M., Gorin, M. A., & Rowe, S. P. (2023). Prostate-specific membrane antigen reporting and data system version 2.0. _European Urology_ , _84_ (5), 491–502. 

https://doi.org/10.1016/j.eururo.2023.06.008 

91 

Wilcoxon, F. (1945). Individual comparisons by ranking methods. _Biometrics Bulletin_ , 

_1_ (6), 80–83. https://doi.org/10.2307/3001968 

Xu, Y., Lin, X., & Zheng, Y. (2024). Poisson ordinal network for Gleason group estimation using bi-parametric MRI. _arXiv preprint arXiv:2407.05796_ . https://doi.org/10.48550/arXiv.2407.05796 

Yan, S., Wang, C., Chen, W., & Lyu, J. (2022). Swin transformer-based GAN for multi-modal medical image translation. _Frontiers in Oncology_ , _12_ . https://doi.org/10.3389/fonc.2022.942511 

Yang, T., Zhang, H., Peng, H., Niu, X., Yang, F., Zhang, J., Wang, Q., Fan, J., Song, Y., & Tao, W. (2026). PSMA PET/MRI-based Swin Transformer architecture for Gleason score prediction in prostate cancer. _Medical Physics_ , _53_ , e70274. https://doi.org/10.1002/mp.70274 

Yoo, S., Gujrathi, I., Haider, M. A., & Khalvati, F. (2019). Prostate cancer detection using deep convolutional neural networks. _Scientific Reports_ , _9_ (1), 19518. https://doi.org/10.1038/s41598-019-55972-4 

Yun, H., Kim, J., Gandhe, A., Nelson, B., Hu, J. C., Gulani, V., Margolis, D., Schackman, B. R., & Jalali, A. (2023). Cost-effectiveness of annual prostate MRI and potential MRI-guided biopsy after prostate-specific antigen test results. _JAMA Network Open_ , _6_ (11), e2344856. https://doi.org/10.1001/jamanetworkopen.2023.44856 

Zhao, X., Liu, S., Zou, Z., & Liang, C. (2025). Global, regional, and national prevalence of prostate cancer from 1990 to 2021: a trend and health inequality analyses. 

92 

_Frontiers in Public Health_ , _13_ , 1595159. 

https://doi.org/10.3389/fpubh.2025.1595159 

Zheng, Y., Zhang, J., Huang, D., Hao, X., Qin, W., & Liu, Y. (2024). Detecting 

MRI-invisible prostate cancers using a weakly supervised deep learning model. 

_International Journal of Biomedical Imaging_ , _2024_ , 2741986. 

https://doi.org/10.1155/2024/2741986 

