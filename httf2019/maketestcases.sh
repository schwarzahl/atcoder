for id in `seq -w 1 30`
do
  python maker.py > testcases/${id}.txt
done
