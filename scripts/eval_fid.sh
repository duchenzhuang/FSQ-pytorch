CUDA_VISIBLE_DEVICES=0 python eval_fid.py --quantizer fsq --levels 8 8 8 5 5 5 --load /public/MARS/Users/dcz/code/vqvae/checkpoints/fsq-n_embed-64000/ckpts/95.pt --batch-size 100 --num-workers 20

# CUDA_VISIBLE_DEVICES=1 python eval_fid.py --quantizer fsq --levels 8 5 5 5 --load /public/MARS/Users/dcz/code/vqvae/checkpoints/fsq-n_embed-1024/ckpts/95.pt --batch-size 100 --num-workers 20
# CUDA_VISIBLE_DEVICES=2 python eval_fid.py --quantizer fsq --levels 7 5 5 5 5 --load /public/MARS/Users/dcz/code/vqvae/checkpoints/fsq-n_embed-4375/ckpts/95.pt --batch-size 100 --num-workers 20
# CUDA_VISIBLE_DEVICES=3 python eval_fid.py --quantizer fsq --levels 8 8 8 6 5 --load /public/MARS/Users/dcz/code/vqvae/checkpoints/fsq-n_embed-15360/ckpts/95.pt --batch-size 100 --num-workers 20
