folder="data"
# 建议加上引号以防文件名有空格
files=($(ls "$folder"))
num_files=${#files[@]}

# 注意：这里原来的 i<1 逻辑看起来不太对，如果是想无限循环，建议使用 while true
# 这里我修正了语法符号，假设你原本是想写 i<1 (只跑一轮)
for ((j=0; j<1; j++)); do
  # 修正 &lt; 为 <
  for ((i=0; i<num_files; i++)); do
    fn="${files[$i]}"

    # 这里的 if [ "$i" -gt -1 ] 恒为真（因为i从0开始），可以直接去掉
    # 如果非要保留，修正 &gt; 为 >
    if [ "$i" -gt -1 ]; then
      deepspeed pretrain.py --data_file "$folder/$fn" --ss $i
    fi
  done
done

