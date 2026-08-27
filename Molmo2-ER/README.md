---
license: apache-2.0
language:
- en
pipeline_tag: image-text-to-text
tags:
- multimodal
- vision-language-model
- embodied-reasoning
- spatial-reasoning
- molmo
- molmo2
base_model:
- allenai/Molmo2-4B
library_name: transformers
---

<img src="assets/Molmo2ER.svg" alt="Molmo2-ER Logo" style="width: auto; height: 50px;">

# Molmo2-ER

**Molmo2-ER** (Embodied Reasoning) is a 4B vision–language model specialized for the embodied perception skills that downstream action models depend on: scene understanding, pixel-accurate pointing, multi-image and egocentric–exocentric correspondence, and video temporal reasoning.

It is built on top of [Molmo2](https://github.com/allenai/molmo2) (Qwen3-4B backbone + SigLIP2 vision encoder) and serves as the vision–language backbone of the **MolmoAct2** action reasoning model.

## Highlights

- **Outperforms every open-weight baseline** as well as the strongest closed-source models — including **Gemini Robot-ER 1.5 Thinking** and **GPT-5** — on **9 of 13** established embodied reasoning benchmarks (Point-Bench, RefSpatial, BLINK, CV-Bench, ERQA, EmbSpatial, MindCube, SAT, VSI-Bench).
- **Overall average 63.8%**, a **+17 point** improvement over the Molmo2 starting point.

## Training

Molmo2-ER is trained from the released Molmo2 checkpoint with a two-stage *specialize-then-rehearse* recipe:

| Stage | Steps | Mixture | Seq. len. | Per-device BS |
|---|---|---|---|---|
| **1. Embodied specialization** | 20K | 3.3M-sample embodied corpus (SAT, RoboPoint, RefSpatial, VST-P, VSI-590K, SIMS-VSI, RoboVQA, SenseNova-SI, CLEVR, GRiD-3D) + 8% Tulu-3 | 4,200 | 4 |
| **2. Joint refinement** | 1.5K | 50% embodied / 42% Molmo2 general / 8% Tulu-3 | 16,384 | 1 |

All other hyperparameters follow [Molmo2](https://github.com/allenai/molmo2).

## Resources

- **Code**: https://github.com/allenai/molmo2
- **Base model**: [Molmo2-4B](https://github.com/allenai/molmo2)

## Usage

See https://github.com/allenai/molmo2 for inference, evaluation, and training code.

## License

Apache-2.0.

## Citation

```bibtex
@misc{fang2026molmoact2actionreasoningmodels,
      title={MolmoAct2: Action Reasoning Models for Real-world Deployment}, 
      author={Haoquan Fang and Jiafei Duan and Donovan Clay and Sam Wang and Shuo Liu and Weikai Huang and Xiang Fan and Wei-Chuan Tsai and Shirui Chen and Yi Ru Wang and Shanli Xing and Jaemin Cho and Jae Sung Park and Ainaz Eftekhar and Peter Sushko and Karen Farley and Angad Wadhwa and Cole Harrison and Winson Han and Ying-Chun Lee and Eli VanderBilt and Rose Hendrix and Suveen Ellawela and Lucas Ngoo and Joyce Chai and Zhongzheng Ren and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2026},
      eprint={2605.02881},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.02881}, 
}
```
