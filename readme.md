# Scalable Language Model with Generalized Continual Learning


## 📢Temporary Statement !!!
Thank you for your interest in our work! We have received many requests regarding our code and decided to release the raw code for our ICLR 2024 paper "*Scalable Language Model with Generalized Continual Learning*". However, please note that we have not yet provided any supplementary explanations. 

I've been occupied lately with another video-generation project. Due to this, we've decided to release all our utilized scripts and codes first. While they may seem a bit disorganized, they are functional. I plan to restructure and tidy them up during the ICLR conference period.


## Suggestions

Although we have completed this work, there are still some shortcomings. If you wish to continue the related working, I will provide some of my own suggestions. May these will help you.
1. **Bert Benchmark.** This benchmark is already approaching its upper limit with existing methods, making it difficult to further improve. Continuing to work on this benchmark will be very difficult.
2. **Llama Benchmark.** We find the fusion of large language models (LLMs) with continual learning to be both intriguing and of considerable practical significance. We conducted experiments on the Llama model to achieve our objectives, yet we acknowledge that the problem definition may not be optimal. As evident, different tasks may present varying levels of difficulty, and we recognize that our initial task may be too simplistic. You are welcome to refine and adjust this setup according to your discretion.
3. **Batch size.** We follow the L2P , which assumes that all queries within the same batch share the same source. However, this may be a trick and could potentially simplify the retrieval process. To address this concern, we mitigate this phenomenon by employing a robust retriever model, ensuring its effectiveness even when the batch size is set to 1. This aspect could also serve as an intriguing point for discussion.

4. **Engineering.** This may not be too research-related, and you can just ignore. I don't think the current way of weight increments is very elegant. Absolutely, exploring methods to save memory and reduce inference costs through engineering techniques is an intriguing pursuit.

## References

This repository owes its existence to the exceptional contributions of other projects. If it weren't for the help of the following work, this job wouldn't have been completed:

* L2P: https://github.com/google-research/l2p
* ProgressivePrompts: https://github.com/arazd/ProgressivePrompts
* Alpaca: https://github.com/tatsu-lab/stanford_alpaca
* Transformers: https://github.com/huggingface/transformers

Many thanks to their invaluable contributions.

<!-- ## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@article{peng2023hierarchical,
  title={Hierarchical Dense Correlation Distillation for Few-Shot Segmentation},
  author={Peng, Bohao and Tian, Zhuotao and Wu, Xiaoyang and Wang, Chenyao and Liu, Shu and Su, Jingyong and Jia, Jiaya},
  journal={arXiv preprint arXiv:2303.14652},
  year={2023}
}
``` -->