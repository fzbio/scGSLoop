#python predict_eval.py 0 3 742 0

for cell_num in 2869;
do
  for ((eval_time=0; eval_time<1; eval_time++));
  do
    python predict_eval.py 0 3 "${cell_num}" "${eval_time}";
  done
done
