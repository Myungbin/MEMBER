# MEMBER

This is the official implementation of MEMBER **(Mixture-of-Experts for Multi-BEhavior Recommendation)** 

[![arXiv](https://img.shields.io/badge/arXiv-2508.19507-b31b1b.svg)](https://arxiv.org/abs/2508.19507)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

(Accepted for ACM CIKM 2025 Full Research Paper)

---

## 📑 Paper & Appendix

- Paper: [arXiv:2508.19507](https://arxiv.org/abs/2508.19507)  
- Accepted at: **ACM CIKM 2025 (Full Research Paper)**  
- Online Appendix: See `MEMBER-Online Appendix.pdf` in the root folder.

---
## 📊 Datasets

We use three widely adopted multi-behavior recommendation datasets:  
- **Tmall**  
- **Taobao**  
- **Jdata**

### Preprocessing
```bash
cd data/{data_name}
python preprocess.py
```

---
## 📈 Evaluation

We report three evaluation results:
1. **Overall performance** under the standard evaluation  
2. **Performance on visited items**  
3. **Performance on unvisited items**
   
### How to Run MEMBER
```bash
cd METHOD
```
* **Tmall**
```bash
python main.py --data_name tmall --con_s 0.1 --temp_s 0.6  --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
```
* **Taobao**
```bash
python main.py --data_name taobao --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6 --device cuda:2 --mask_validation
```
* **Jdata**
```bash
python main.py --data_name jdata --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
```

### Training With Data Variants
If you want to train on a subset variant stored under `data_variants/{data_name}/{variant_name}`, pass `--data_variant`.

```bash
python main.py --data_name tmall --data_variant keep_cart_buy --device cuda:0 --mask_validation
```

`metadata.json` inside the variant directory is used to resolve the active behavior list automatically, so variants such as `keep_buy`, `keep_cart_buy`, and `keep_view_collect_cart_buy` can be trained without editing the code.

When `--model_name` is omitted, logs and checkpoints are named automatically with a variant-aware experiment name such as `tmall_keep_cart_buy`, so each run is easy to distinguish.

If you want to point to a custom preprocessed dataset directory directly, use `--data_path` instead of `--data_variant`.

To run every available variant for a dataset sequentially:

```bash
python run_all_variants.py --datasets tmall -- --device cuda:0 --mask_validation
```

Use `--dry_run` first if you want to inspect the generated commands without executing them.

To generate commands with the same per-dataset hyperparameters used in this README:

```bash
python run_all_variants.py --preset readme --dry_run
```

An explicit command list for every current variant is also available in [README_VARIANT_COMMANDS.md](/home/mbgwak/workspace/MEMBER/README_VARIANT_COMMANDS.md).

## 📚 Citation
If you find MEMBER useful, please cite our paper:
```bibtex
@inproceedings{kim2025self,
  title     = {A Self-Supervised Mixture of Experts Framework for Multi-behavior Recommendation},
  author    = {Kim, Kyungho and Kim, Sunwoo and Lee, Geon and Shin, Kijung},
  booktitle = {CIKM},
  year      = {2025}
}
