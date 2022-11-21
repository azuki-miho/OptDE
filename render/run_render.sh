REALDATA=ModelNet
RCLASS=sofa
LOGDATE=Log_2021-08-31_09-26-13
RESULT_NAME=best_results
FINETUNE=finetune_
LOGDIR=logs
for IDX in 0 1
do
    python3 render_mitsuba2_pc.py --dataset ${REALDATA} --category ${RCLASS} --log_date ${LOGDATE} --result_name ${RESULT_NAME} --finetune ${FINETUNE} --idx ${IDX} --log_dir ${LOGDIR}
done
