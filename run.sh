VIRTUALDATA=CRN
REALDATA=3D_FUTURE
VCLASS=chair
RCLASS=chair
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python trainer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--class_choice ${VCLASS} \
--split train \
--epoch 200 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${REALDATA}_${RCLASS} \
--ckpt_load pretrained_models/${VCLASS}.pt \
--dataset_path ./datasets/our_data/ \
--log_dir ${LOGDIR}
