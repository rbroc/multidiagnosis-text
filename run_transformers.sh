#!/bin/bash
for model in vesteinn/ScandiBERT xlm-roberta-base sentence-transformers/distiluse-base-multilingual-cased-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 Maltehb/aelaectra-danish-electra-small-cased
do 
    for lr in 0.01 0.001 0.0001 0.00001 0.000001
    do 
        for batch in 16
        do
            for warmup in 100
            do
                for fl in 0 1
                do  
                    mid="$(cut -d'/' -f2 <<<"$model")"
                    echo $mid
                    python3 train_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl
                done
            done
        done
    done
done

for bnr in ASD DEPR SCHZ
do
    for model in vesteinn/ScandiBERT xlm-roberta-base sentence-transformers/distiluse-base-multilingual-cased-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 Maltehb/aelaectra-danish-electra-small-cased
    do 
        for lr in 0.01 0.001 0.0001 0.00001 0.000001
        do 
            for batch in 16
            do
                for warmup in 100
                do
                    for fl in 0 1
                    do  
                        mid="$(cut -d'/' -f2 <<<"$model")"
                        echo $mid
                        python3 train_transformer.py --model-id $mid --checkpoint $model --epochs 100 --train-examples-per-device $batch --eval-examples-per-device $batch --logging-steps 1000 --learning-rate $lr --weight-decay 0.001 --warmup-steps $warmup --freeze-layers $fl --binary $bnr
                    done
                done
            done
        done
    done
done
