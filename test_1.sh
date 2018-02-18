for num in 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000; do
    python translate.py --decode --model translate.ckpt-${num} < ../data_iwslt/test.en > res${num}
    echo "After \"${num}\" updates, the BLEU is:"
    perl multi-bleu.perl ../data_iwslt/devtest/devset3.lc.en < res${num}
done
