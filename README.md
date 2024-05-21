## Training & Testing

### Training

```
python main.py --half=True --batch_size=128 --test_batch_size=128 \
    --step 35 55 --num_epoch=65 --n_heads=3 --num_worker=4 --k=1 \
    --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=0.05 --z_prior_gain=3 \
    --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 \
    --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder
```

- To ensemble the results of different modalities, run the following command:
```
python ensemble.py \
   --dataset=ntu/xsub \
   --position_ckpts \
      <work_dir_1>/files/best_score.pkl \
      <work_dir_2>/files/best_score.pkl \
      ...
   --motion_ckpts \
      <work_dir_3>/files/best_score.pkl \
      <work_dir_4>/files/best_score.pkl \
      ...
```
