# LoCoMo dataset attribution

`locomo10.json` is inherited unchanged from the pinned A-Mem baseline. Its original source is the [LoCoMo repository](https://github.com/snap-research/locomo), accompanying **Evaluating Very Long-Term Conversational Memory of LLM Agents** (ACL 2024) by Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang.

LoCoMo is released under [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE-LOCOMO.txt). The full license was copied from the [official license file](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) and also applies to dataset subsets and question/answer excerpts in benchmark reports. Project code uses the MIT license.

The file's SHA256 is `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. It matches the official LoCoMo `main` dataset checked on 2026-10-18. Experiments select chronological turns and source question indices. Reports store generated answers alongside the original references.

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```
