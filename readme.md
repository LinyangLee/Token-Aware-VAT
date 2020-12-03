# Token-Aware Virtual Adversarial Training

Code for our AAAI 2021 paper :



*[TAVAT: Token-Aware Virtual Adversarial Training for Language Understanding
](https://arxiv.org/abs/2004.14543)*


## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.4.0
- [transformers](https://github.com/huggingface/transformers) 2.9.0


## Usage

First please use pip to install pytorch and hugging-face transformers 2.9.0.

The token_vat script is modified from run_glue.py in the transformers repository.

The hyper-parameters are mostly copied from [FreeLB](https://github.com/zhuchen03/FreeLB) with minor modifications.


Run RTE task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 2e-5 --do_train --task_name rte --data_dir data/RTE/ --output_dir outputs/rte_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 50 --logging_steps 50 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 30 --num_train_epochs 9 --adv_lr 2e-2 --adv_init_mag 1.6e-1 --adv_max_norm 1.4e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```

Run CoLA task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 3e-5 --do_train --task_name cola --data_dir data/CoLA/ --output_dir outputs/cola_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 100 --logging_steps 100 --evaluate_during_training --per_gpu_train_batch_size 16 --warmup_steps 30 --num_train_epochs 12 --adv_lr 2e-2 --adv_init_mag 1e-1 --adv_max_norm 2e-1 --adv_steps 3 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```

Run MRPC task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 2e-5 --do_train --task_name mrpc --data_dir data/MRPC/ --output_dir outputs/mrpc_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 100 --logging_steps 100 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 30 --num_train_epochs 9 --adv_lr 5e-2 --adv_init_mag 1e-1 --adv_max_norm 2e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```

Run STS task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 2e-5 --do_train --task_name sts-b --data_dir data/STS-B/ --output_dir outputs/sts_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 100 --logging_steps 100 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 30 --num_train_epochs 9 --adv_lr 1e-1 --adv_init_mag 1e-1 --adv_max_norm 2.8e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```

Run SST-2 task

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 1e-5 --do_train --task_name sst-2 --data_dir data/sst-2/ --output_dir outputs/sst_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 1000 --logging_steps 1000 --evaluate_during_training --per_gpu_train_batch_size 16 --warmup_steps 500 --num_train_epochs 9 --adv_lr 5e-2 --adv_init_mag 2e-1 --adv_max_norm 5e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```


Run QNLI task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 1e-5 --do_train --task_name qnli --data_dir data/QNLI/ --output_dir outputs/qnli_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 500 --logging_steps 500 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 2000 --num_train_epochs 9 --adv_lr 2e-2 --adv_init_mag 1e-1 --adv_max_norm 3e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```



Run QQP task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 2e-5 --do_train --task_name qqp --data_dir data/QQP/ --output_dir outputs/qqp_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 5000 --logging_steps 5000 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 5000 --num_train_epochs 9 --adv_lr 2e-1 --adv_init_mag 3.2e-1 --adv_max_norm 7e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```


Run MNLI task 

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python token_vat.py --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --learning_rate 3e-5 --do_train --task_name mnli --data_dir data/MNLI/ --output_dir outputs/mnli_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 5000 --logging_steps 5000 --evaluate_during_training --per_gpu_train_batch_size 32 --warmup_steps 5000 --num_train_epochs 9 --adv_lr 1e-1 --adv_init_mag 3e-1 --adv_max_norm 4e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1
```



## Note 

this repository is still in progress.