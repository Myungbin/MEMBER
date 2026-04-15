# Variant Commands

These commands use the same per-dataset hyperparameters shown in `README.md`.

## Tmall

```bash
python main.py --data_name tmall --data_variant keep_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_click_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_click_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_click_collect_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_click_collect_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_collect_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
python main.py --data_name tmall --data_variant keep_collect_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2 --device cuda:1 --mask_validation
```

## Taobao

```bash
python main.py --data_name taobao --data_variant keep_buy --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6 --device cuda:2 --mask_validation
python main.py --data_name taobao --data_variant keep_cart_buy --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6 --device cuda:2 --mask_validation
python main.py --data_name taobao --data_variant keep_view_buy --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6 --device cuda:2 --mask_validation
python main.py --data_name taobao --data_variant keep_view_cart_buy --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6 --device cuda:2 --mask_validation
```

## Jdata

```bash
python main.py --data_name jdata --data_variant keep_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_collect_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_collect_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_view_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_view_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_view_collect_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:3 --mask_validation
python main.py --data_name jdata --data_variant keep_view_collect_cart_buy --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2 --device cuda:1
```

## Auto-Generate

You can regenerate the same commands with:

```bash
python run_all_variants.py --preset readme --dry_run
```

```bash
python run_all_variants.py --datasets tmall --preset readme -- --device cuda:0
python run_all_variants.py --datasets taobao --preset readme -- --device cuda:1
python run_all_variants.py --datasets jdata --preset readme -- --device cuda:1
```