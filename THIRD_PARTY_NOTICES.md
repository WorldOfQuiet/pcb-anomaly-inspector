# Third-party resources

The command-line application code is provided under the MIT license. The following
data and model components retain their own terms and attribution.

| Resource | Attribution and terms |
| --- | --- |
| VisA / PCB1 images | Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV 2022. [Source](https://github.com/amazon-science/spot-diff), revision `2a692ab575001cbde74d402d897a7286086c6199`. Dataset license: [CC BY 4.0](licenses/VisA-CC-BY-4.0.txt). Three development JPEGs are included without changing their bytes; only filenames changed. Source names and hashes are in [examples/provenance.json](examples/provenance.json). |
| PatchCore | Karsten Roth et al., *Towards Total Recall in Industrial Anomaly Detection*. This project uses [anomalib](https://github.com/open-edge-platform/anomalib) 2.6.2, [Apache-2.0](licenses/Apache-2.0.txt). The PatchCore implementation is installed as a dependency, not represented as original project work. |
| ResNet18 IMAGENET1K_V1 | Weights provided by [PyTorch](https://download.pytorch.org/models/resnet18-f37072fd.pth), SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. [Torchvision](https://github.com/pytorch/vision) code is [BSD-3-Clause](licenses/torchvision-BSD.txt). That code license does not grant rights to ImageNet source images; none are distributed here. |
| Frozen `pcb1-v1` checkpoint | A PatchCore memory bank fitted on 604 normal VisA/PCB1 images with an ImageNet ResNet18 feature extractor. The artifact derives from the resources above. Its training provenance, immutable weights hash, split protocol, and source metadata hashes are in [models/pcb1-v1/provenance.json](models/pcb1-v1/provenance.json). |

Dependency license texts also accompany their installed distributions. No employer
data, employer branding, or endorsement is included.
